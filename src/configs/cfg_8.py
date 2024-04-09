from types import SimpleNamespace
import torch
import socket

import src.utils.augs as A

cfg = SimpleNamespace(**{})
cfg.data_dir= "./data/"
cfg.project= "hms"
cfg.weights_path= ""
cfg.hostname = socket.gethostname()
cfg.device= torch.device("cuda" if torch.cuda.is_available() else "cpu")
cfg.fast_dev_run= False
cfg.no_wandb= True
cfg.save_weights= True
cfg.save_preds= True
cfg.val_mode= False
cfg.train_all_data= True
cfg.seed= -1
cfg.val_fold= 0
cfg.pretrain= True
cfg.metadata= "train_25.0sec_nooverlap.csv"

# Optimizer + Scheduler
cfg.scheduler = "CosineAnnealingLR"
cfg.lr = 1e-4
cfg.batch_size= 16
cfg.weight_decay = 1e-4
cfg.epochs= 14
cfg.swa= False
cfg.swa_epochs= 2
cfg.ema= False
cfg.ema_decay= 0.99
cfg.label_smoothing= 0.0

# Model
cfg.encoder_model_type= "milAllbfg"
cfg.backbone_comp= "tf_efficientnetv2_b0.in1k"
cfg.backbone= "tf_efficientnetv2_s.in21k_ft_in1k"

encoder_config = SimpleNamespace(**{})
encoder_config.input_dropout_p= 0.2 # AKA drop_path_rate
encoder_config.fully_connected_dropout_p= 0.5
encoder_config.attention_dropout_p= 0.5
encoder_config.attention_emb_dim= 512
cfg.encoder_config = encoder_config

# STFTs on the fly (ignores 2D augs)
stft_config = SimpleNamespace(**{})
stft_config.stft_n_fft=190       # height
stft_config.stft_win_length=96   # width
stft_config.stft_hop_length=24   # width
stft_config.stft_trainable=True
stft_config.stft_stack_factor= 2
stft_config.stft_fmax= 40.0
stft_config.stft_window= "hann"
stft_config.stft_freq_scale= "linear"
cfg.stft_config= stft_config

# WaveNet
cfg.seq_len= 10_000
cfg.diffs= ["Bipolar 2xA", "Bipolar 2xB", "Transverse"]
cfg.diff_idxs= [(0,8,8),(8,16,8),(16,24,8)] # start, end, len
cfg.drop_n_patients= 0
cfg.wv_n_blocks= 8
cfg.wv_out_channels= 28
cfg.wv_ds_factor= 5

# PL Trainer
cfg.accelerator= "gpu"
cfg.benchmark= True # set to True if input size does not change (increases speed)
cfg.num_sanity_val_steps= 0
cfg.overfit_batches= 0
cfg.precision= "16-mixed"
cfg.accumulate_grad_batches= 1
cfg.enable_checkpointing= False
cfg.gradient_clip_val= 1.0

# Dataset/Dataloader
cfg.pin_memory= True
cfg.num_workers= 5

# Augs
cfg.n_mixup= 2
cfg.train_aug_2d= None
cfg.val_aug_2d= None
cfg.butter_cutoff_freq= 50
cfg.butter_order= 4
cfg.gauss_noise_sigma= 0.015
cfg.head_scale_range= 0.075
cfg.noisy_student_prob= 0.15
cfg.fb_flip_prob= 0.5
cfg.lr_flip_prob= 0.5
cfg.shuffle_ds_prob= 0.2

cfg.val_offset= 0
cfg.val_lr_flip= False
cfg.val_fb_flip= False


# ----- 2D Aug: Comp spectrograms -----
cfg.train_aug_2d_comps= A.Compose([
        # -- Numpy Augs --
        A.NormFillNAN(p=1.0),
        A.CoarseDropout(p=0.5, min_holes=1, max_holes=10, min_height=1, max_height=10, min_width=1, max_width=10, fill_value=0),
        A.OneOf([
            A.MotionBlur(blur_limit=3, p=0.5),
            A.MedianBlur(blur_limit=3, p=0.5),
        ], p=0.3),
        A.NumpyToTorch(p=1.0),
        # -- Torch Augs --
        A.TemporalResample(p=1.0, sample_rate=(0.85, 1.05)),
        A.TemporalPad(p=1.0, length=300),
        A.TemporalCenterCrop(p=1.0, length=300),
        A.FrequencyDropout(p=0.5, max_dropout=5),
    ])
cfg.train_aug_2d_comps._disable_check_args() # disable type check otherwise input must be numpy/ int8

cfg.val_aug_2d_comps= A.Compose([
    A.NormFillNAN(p=1.0),
    A.NumpyToTorch(p=1.0),
    A.TemporalCenterCrop(p=1.0, length=300),
])
cfg.val_aug_2d_comps._disable_check_args()

# --- 1d Augs ---
cfg.train_aug_1d= A.Compose([
    A.NumpyToTorch(p=1.0),
])
cfg.train_aug_1d._disable_check_args() # disable type check otherwise input must be numpy/ int8

cfg.val_aug_1d= A.Compose([
    A.NumpyToTorch(p=1.0),
])
cfg.val_aug_1d._disable_check_args()