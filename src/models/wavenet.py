from typing import List

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from .utils import count_parameters

class WaveBlock(nn.Module):
    def __init__(self, in_channels, filters, kernel_size, n, downsample=None):
        super().__init__()
        self.dilation_rates = [2**i for i in range(n)]
        self.filters= filters
        
        self.conv1x1 = nn.Conv1d(in_channels, filters, kernel_size=1, padding=0)

        dil_convs = []
        skip_convs = []
        for dilation_rate in self.dilation_rates:
            padding = int((dilation_rate * (kernel_size - 1)) / 2) # Same padding

            dil_convs.extend([
                nn.Conv1d(filters, filters, kernel_size=kernel_size, padding=padding, dilation=dilation_rate),
                nn.Conv1d(filters, filters, kernel_size=kernel_size, padding=padding, dilation=dilation_rate),
                nn.Conv1d(filters, filters, kernel_size=kernel_size, padding=padding, dilation=dilation_rate),
            ])

            skip_convs.extend([
                nn.Conv1d(filters, filters, kernel_size=1, padding=0),
            ])
        
        self.skip_convs = nn.ModuleList(skip_convs)
        self.dil_convs = nn.ModuleList(dil_convs)

        if downsample: 
            self.ds_conv = nn.Conv1d(filters, filters, kernel_size=downsample, stride=downsample, padding=0)
        else:
            self.ds_conv= None

        self.act_tan= nn.Tanh()
        self.act_sig= nn.Sigmoid()
        self.act_mish= nn.Mish()
        
    def forward(self, x):
        x = self.conv1x1(x)
        res_x = x
        for i, s_conv in enumerate(self.skip_convs):
            x1= self.dil_convs[(i*3)](x)
            x2= self.dil_convs[(i*3)+1](x)
            x3= self.dil_convs[(i*3)+2](x)
            x = self.act_sig(x1) * self.act_tan(x2) * self.act_mish(x3)
            x = s_conv(x)
            x = x + res_x
        if self.ds_conv:
            x= self.ds_conv(x)
        return x
    
class WaveNet(nn.Module):
    def __init__(self, cfg, features_only=False):
        super().__init__()
        self.cfg = cfg
        self.num_classes = 6
        self.features_only = features_only

        # Variable blocks
        in_chans= list(np.linspace(1, self.cfg.wv_out_channels, self.cfg.wv_n_blocks).astype(int))
        out_chans= in_chans[1:] + [self.cfg.wv_out_channels]
        dil_rates= list(np.linspace(15, 1, self.cfg.wv_n_blocks).astype(int))
        ds_factors= [None]*(self.cfg.wv_n_blocks-1) + [self.cfg.wv_ds_factor]

        blocks= []
        for i in range(self.cfg.wv_n_blocks):
            blocks.append(
                WaveBlock(
                    in_channels= in_chans[i], 
                    filters= out_chans[i], 
                    kernel_size= 3, 
                    n= dil_rates[i], 
                    downsample= ds_factors[i],
                    )
            )
            print(in_chans[i], out_chans[i], 3, dil_rates[i], ds_factors[i])
        self.wave_blocks= nn.ModuleList(blocks)

        if self.features_only:
            self.pool = nn.Identity()
            self.linear = nn.Identity()
        else:
            self.pool = nn.AdaptiveMaxPool1d(1) # AVG pooling
            self.linear = nn.Linear(self.wave_blocks[-1].residual_add.out_channels, self.num_classes)
            print('n_params:',count_parameters(self))

    def forward(self, x):
        # x = x.pop("input_eeg_raw")
        b, c, t = x.size()

        # Wavenet blocks
        for wb in self.wave_blocks:
            x = wb(x)

        # Pooling
        x = self.pool(x)
        x = x.permute(0,2,1)
        x = self.linear(x).squeeze(1)
        return x

class WaveNetENHeadV5(nn.Module):
    def __init__(self, cfg, features_only=False):
        super().__init__()
        self.cfg = cfg
        self.num_classes= 6
        self.seq_len= self.cfg.seq_len

        # 1D
        self.backbone_1ds= nn.ModuleList([
            WaveNet(cfg=cfg, features_only=True) for i in range(len(self.cfg.diffs))
            ])

        # 2D
        self.features_only= features_only
        self.backbone_2d= timm.create_model(
            model_name= cfg.backbone,
            pretrained= True,
            num_classes= self.num_classes,
            drop_path_rate= cfg.encoder_config.input_dropout_p,
            in_chans= 1,
        )
        self.backbone_2d.set_grad_checkpointing()
        
        # ConvStem + num_features
        if "hgnetv2" in self.cfg.backbone:
            self.num_features= self.backbone_2d.head.fc.in_features
            self.backbone_2d.head.fc= nn.Identity()
        else:
            self.backbone_2d.conv_stem.stride= (1,2)
            self.backbone_2d.classifier= nn.Identity()
            self.num_features= self.backbone_2d.num_features

        # Head
        self.fc_dropout = nn.Dropout(p= 0.1)
        if not self.features_only:
            self.fc = nn.Linear(self.num_features, self.num_classes)
            print('n_params:',count_parameters(self))
        else:
            self.fc = nn.Identity()

    def forward(self, x, cropcat_idxs=torch.tensor([]), half_seg=False):
        if isinstance(x, dict): x = next(iter(x.values()))
        b, c, t= x.size()

        # One backbone per 8 diffs
        arr = []
        for i, (bb_1d, diff_idx) in enumerate(zip(self.backbone_1ds, self.cfg.diff_idxs)):
            start, end, d_len= diff_idx
            x_cur= x[:, start:end, :]
            x_cur= x_cur.contiguous().view(b*d_len, self.seq_len).unsqueeze(1)
            x_cur = bb_1d(x_cur)
            x_cur = x_cur.view(b, d_len, x_cur.shape[1], self.cfg.wv_out_channels)
            arr.append(x_cur)
        
        # Creates small + large feature stack (eg. So multi-node features fall in 1 receptive field)
        x2 = torch.cat([x[..., :-2] for x in arr], dim=1)
        x3 = torch.cat([x[..., -2:] for x in arr], dim=1)

        # Combine
        x2 = x2.permute(0, 2, 1, 3).contiguous().view(b, x2.shape[2], -1)
        x3 = x3.permute(0, 2, 1, 3).contiguous().view(b, x3.shape[2], -1)
        x = torch.cat([x2, x3], dim=2).unsqueeze(1)

        # 2D Backbone
        x = self.backbone_2d(x)
        x = self.fc_dropout(x)
        x = self.fc(x)
        return x

if __name__ == "__main__":
    import torchinfo

    from src.configs.cfg_1 import cfg
    x = torch.ones(1, 24, 10_000)
    model = WaveNetENHeadV5(cfg=cfg, features_only=True)
    z = model(x)
    print(z.shape)