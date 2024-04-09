import torch
import torch.nn as nn
import timm
from types import SimpleNamespace

from src.models.spect import STFTLayer, SpectStackClassifier
from src.models.wavenet import WaveNetENHeadV5

from .utils import count_parameters

class MILModel(nn.Module):
    """
    MIL Model w/ self attention.

    Source: https://arxiv.org/pdf/1703.03130.pdf
    """
    def __init__(self, cfg: SimpleNamespace, features_only: bool = False, num_classes: int = 6):
        super().__init__()
        self.num_classes= num_classes
        self.cfg= cfg
        self.features_only= features_only
        self.backbone = timm.create_model(
            model_name= cfg.backbone_comp,
            pretrained= True,
            num_classes= self.num_classes,
            drop_path_rate= cfg.encoder_config.input_dropout_p,
            in_chans= 1,
        )
        self.backbone.set_grad_checkpointing()

        self.fc_dropout = nn.Dropout(p= cfg.encoder_config.fully_connected_dropout_p)
        self.att_dropout = nn.Dropout(p= cfg.encoder_config.attention_dropout_p)
        
        if "hgnetv2" in self.cfg.backbone_comp:
            self.backbone.num_features= self.backbone.head.fc.in_features
            self.backbone.head.fc= nn.Identity()
        else:
            self.backbone.classifier= nn.Identity()
        self.attention = nn.Sequential(nn.Linear(self.backbone.num_features, 2048), nn.Tanh(), nn.Linear(2048, 1))

        if not self.features_only:
            self.fc = nn.Linear(self.backbone.num_features, self.num_classes)
        else:
            self.fc = nn.Identity()
        self.num_features = self.backbone.num_features

    def forward(self, x):
        if isinstance(x, dict): x = next(iter(x.values()))

        b, t, c, h, w = x.size()

        # Backbone
        x = x.view(b*t, c, h, w)
        x = self.backbone(x)
        x = self.fc_dropout(x)
        x = x.view(b, t, -1)

        # Attention
        a = self.attention(x)
        a = torch.softmax(a, dim=1)
        a = self.att_dropout(a)
        x = torch.sum(x * a, dim=1)

        x = self.fc(x)
        return x
    
class DoubleMILModel(nn.Module):
    def __init__(self, cfg: SimpleNamespace):
        super().__init__()
        self.cfg = cfg
        self.num_classes = 6

        # Backbones
        self.spect_backbone= MILModel(cfg=cfg, features_only=True)
        self.eeg_backbone= SpectStackClassifier(cfg=cfg, features_only=True)

        # Head
        total_features= self.spect_backbone.backbone.num_features + \
                        self.eeg_backbone.backbone.num_features
        
        self.head_fc1= nn.Linear(total_features, self.num_classes)

    def forward(self, x, cropcat_idxs=torch.tensor([]), half_seg=False):

        # Spectrogram
        x_spec = x["input_comp_spectrogram"].unsqueeze(2)
        x_spec = self.spect_backbone.forward(x_spec)

        # EEG
        x_eeg = x["input_eeg_raw"]
        x_eeg = self.eeg_backbone.forward(x_eeg, cropcat_idxs=cropcat_idxs, half_seg=half_seg)

        # Concat Embeddings
        x_out = torch.concat([x_eeg, x_spec], dim=1)

        # Head
        x_out = self.head_fc1(x_out)

        return x_out

class MILModelV14(nn.Module):
    def __init__(self, cfg: SimpleNamespace):
        super().__init__()
        self.cfg = cfg
        self.num_classes = 6

        # 2D Backbones
        self.spect_backbone= MILModel(cfg=cfg, features_only=True)
        self.eeg_backbone= SpectStackClassifier(cfg=cfg, features_only=True)

        # 1D backbone
        self.eeg_backbone_1d= WaveNetENHeadV5(cfg=cfg, features_only=True)

        # Head
        total_features= self.spect_backbone.backbone.num_features + \
                        self.eeg_backbone.num_features + \
                        self.eeg_backbone_1d.num_features

        # Aux Heads
        self.head_spect= nn.Linear(self.spect_backbone.num_features, self.num_classes)
        self.head_eeg= nn.Linear(self.eeg_backbone.num_features, self.num_classes)
        self.head_1d= nn.Linear(self.eeg_backbone_1d.num_features, self.num_classes)

        self.head_all= nn.Linear(total_features, self.num_classes)
        print('n_params:', count_parameters(self))

    def forward(self, x, cropcat_idxs=torch.tensor([]), half_seg=False):

        # Competition Spectrogram
        x_spec = x["input_comp_spectrogram"].unsqueeze(2)
        x_spec = self.spect_backbone.forward(x_spec)

        # EEG 2d
        x_eeg = x["input_eeg_raw"]
        x_eeg = self.eeg_backbone.forward(x_eeg, cropcat_idxs=cropcat_idxs, half_seg=half_seg)

        # EEG 1d
        x_eeg_1d = x["input_eeg_raw"]
        x_eeg_1d = self.eeg_backbone_1d(x_eeg_1d)

        # Concat Embeddings
        x_out = torch.concat([x_eeg, x_spec, x_eeg_1d], dim=1)

        # Head
        x_out = self.head_all(x_out)

        # Aux Heads
        x_spec= self.head_spect(x_spec)
        x_eeg= self.head_eeg(x_eeg)
        x_eeg_1d= self.head_1d(x_eeg_1d) 

        return x_out, x_spec, x_eeg, x_eeg_1d

if __name__ == "__main__":
    pass