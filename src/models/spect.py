import timm
import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy

from nnAudio.features.stft import STFT

from types import SimpleNamespace

from .utils import count_parameters

class STFTLayer(nn.Module):
    """
    STFT Layer.

    Docs: https://kinwaicheuk.github.io/nnAudio/_autosummary/nnAudio.features.stft.STFT.html?highlight=stft
    """
    def __init__(self, n_fft=512, win_length=128, hop_length=32, fmax=50.0, freq_scale="linear", window="hann", trainable=False):
        super().__init__()        
        self.spec_layer= STFT(
            # ----- Fixed -----
            sr=200, 
            fmin=0.0, 
            center=False,
            verbose=False,
            output_format="Magnitude",
            # ----- Variable -----
            window=window,
            fmax=fmax, 
            freq_scale=freq_scale,
            hop_length=hop_length, 
            n_fft=n_fft,
            win_length=win_length,
            trainable=trainable,
            )
        print("EEG_SPECT_SHAPE: ", self.forward(torch.ones(1, 10_000)).shape)
        
    def forward(self, x):
        # Pass through layer
        x = self.spec_layer(x)

        # Scale + Standardize
        x= torch.log1p(x.pow(2))
        x= torch.clamp(x, min=0.0, max=20.0)
        x= (x - 7.5) / 7.5
        return x

class SpectStackClassifier(nn.Module):
    def __init__(self, cfg: SimpleNamespace, features_only: bool = False, num_classes: int = 6):
        super().__init__()
        self.num_classes= num_classes
        self.cfg= cfg
        self.stft_cfg= self.cfg.stft_config
        self.features_only= features_only
        self.backbone = timm.create_model(
            model_name= cfg.backbone,
            pretrained= True,
            num_classes= self.num_classes,
            drop_path_rate= cfg.encoder_config.input_dropout_p,
            in_chans= 1,
        )
        self.backbone.set_grad_checkpointing()

        self.stft_layer= STFTLayer(
            n_fft= self.stft_cfg.stft_n_fft, 
            win_length= self.stft_cfg.stft_win_length, 
            hop_length= self.stft_cfg.stft_hop_length, 
            fmax= self.stft_cfg.stft_fmax,
            freq_scale= self.stft_cfg.stft_freq_scale,
            window= self.stft_cfg.stft_window,
            trainable= self.stft_cfg.stft_trainable,
        )
        self.stack_factor= self.stft_cfg.stft_stack_factor

        self.fc_dropout = nn.Dropout(p= cfg.encoder_config.fully_connected_dropout_p)
        
        if "hgnetv2" in self.cfg.backbone:
            self.backbone.num_features= self.backbone.head.fc.in_features
            self.backbone.head.fc= nn.Identity()
        else:
            self.backbone.classifier= nn.Identity()

        if not self.features_only:
            self.fc = nn.Linear(self.backbone.num_features, self.num_classes)
            print('n_params:',count_parameters(self))
        else:
            self.fc = nn.Identity()
        self.num_features = self.backbone.num_features

    def forward(self, x, cropcat_idxs=torch.tensor([]), half_seg=False):
        if isinstance(x, dict): x = next(iter(x.values()))
        b, d, seq_len = x.size()
        x= x.view(b*d, seq_len)
        x= self.stft_layer(x)

        # CropCat/Mixup Aug
        x = x.view(b, 1, d*(x.shape[-2]), x.shape[-1])
        if cropcat_idxs.size()[0] > 0:
            n_cropcat= cropcat_idxs.size()[0]
            seq_len= x.shape[3]

            if half_seg:
                mid_idx= seq_len//2
                x[:n_cropcat, :, :, mid_idx:] = x[cropcat_idxs, :, :, mid_idx:]

            else:
                x_a= x[cropcat_idxs, :, :, (seq_len//4):(seq_len//4)*2]
                x_b= x[cropcat_idxs, :, :, (seq_len//4)*3:]
                x[:n_cropcat, :, :, (seq_len//4):(seq_len//4)*2] = x_a
                x[:n_cropcat, :, :, (seq_len//4)*3:] = x_b
        x = x.view(b, 1, x.shape[-2]//self.stack_factor, x.shape[-1]*self.stack_factor)

        # Backbone
        x = self.backbone(x)
        x = self.fc_dropout(x)
        x = self.fc(x)
        return x

if __name__ == "__main__":

    from src.configs.cfg_my_specv2 import cfg

    model = SpectStackClassifier(cfg=cfg)
    x= torch.rand(1,2,10_000)
    z= model(x)
    print(z.shape)