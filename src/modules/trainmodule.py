import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchinfo
import os
import numpy as np
import timm

import json
from types import SimpleNamespace
import wandb

from src.utils.loss import KLDivLossWithLogits, WeightedKLDivWithLogitsLoss
from src.modules.utils import ModelEMA, ModelSWA

class CustomTrainModule(pl.LightningModule):
    def __init__(
        self,
        cfg,
    ):
        super().__init__()
        self.cfg = cfg
        self.model = self._init_model()
        if self.cfg.ema:
            self.ema_model= ModelEMA(self.model, device=self.cfg.device, decay=self.cfg.ema_decay)
        if self.cfg.swa:
            self.swa_model= ModelSWA(self.model, device=self.cfg.device)

        self.loss_fn = self._init_loss_fn()
        self.val_outputs = []

    def _init_model(self):
        if self.cfg.encoder_model_type == "milAllbfg":
            from src.models.mil import MILModelV14
            model = MILModelV14(cfg=self.cfg)
            
        else:
            raise ValueError("Invalid model_type: {}".format(self.cfg.encoder_model_type))

        if self.cfg.weights_path != "": 
            model.load_state_dict(torch.load(self.cfg.weights_path, map_location="cuda:0"))
            print("Loaded pre-trained weights..")
            # print(model.fc.weight)
            
        # torchinfo.summary(model)
        return model
    
    def _init_optimizer(self):
        return optim.AdamW(
            self.trainer.model.parameters(), 
            lr=self.cfg.lr, 
            weight_decay=self.cfg.weight_decay,
            )

    def _init_scheduler(self, optimizer):
        if self.cfg.scheduler == "Constant":
            # Hacky fix to keep constant LR
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, 
                T_max = self.cfg.epochs,
                eta_min = self.cfg.lr,
                )
        elif self.cfg.scheduler == "CosineAnnealingLR":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, 
                T_max = self.cfg.epochs,
                eta_min = 1e-7,
                )
        else:
            raise ValueError(f"{self.cfg.scheduler} is not a valid scheduler.")
        
    def lr_scheduler_step(self, scheduler, optimizer_idx) -> None:
        scheduler.step()
        return
    
    def _init_loss_fn(self):
        return WeightedKLDivWithLogitsLoss()

    def configure_optimizers(self):
        optimizer = self._init_optimizer()
        scheduler = self._init_scheduler(optimizer)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
    
    def cropcat(self, batch, labels, total_votes):
        # CropCat params
        n_cropcat= self.cfg.n_mixup
        cropcat_idxs= torch.randperm(labels.size(0))[:n_cropcat]

        # Transform Labels
        labels[:n_cropcat]= (labels[:n_cropcat] + labels[cropcat_idxs])/2
        total_votes[:n_cropcat]= (total_votes[:n_cropcat] + total_votes[cropcat_idxs])/2

        # 1/2 segments or 1/4 segments
        if np.random.random() > 0.8:
            half_seg= True
        else:
            half_seg= False

        # 1D Data
        if "input_eeg_raw" in batch:
            seq_len= batch["input_eeg_raw"].shape[2]
            
            if half_seg:
                mid_idx= seq_len//2
                batch["input_eeg_raw"][:n_cropcat, :, mid_idx:] = batch["input_eeg_raw"][cropcat_idxs, :, mid_idx:]
            else:
                a= batch["input_eeg_raw"][cropcat_idxs, :, (seq_len//4):(seq_len//4)*2]
                b= batch["input_eeg_raw"][cropcat_idxs, :, (seq_len//4)*3:]
                batch["input_eeg_raw"][:n_cropcat, :, (seq_len//4):(seq_len//4)*2] = a
                batch["input_eeg_raw"][:n_cropcat, :, (seq_len//4)*3:] = b

        # Spectrograms
        for feat in ["input_comp_spectrogram", "input_eeg_spectrogram"]:
            if feat in batch:
                seq_len= batch[feat].shape[2]

                if half_seg:
                    mid_idx= batch[feat].shape[2]//2
                    batch[feat][:n_cropcat, :, mid_idx:, :] = batch[feat][cropcat_idxs, :, mid_idx:, :]
                else:
                    a= batch[feat][cropcat_idxs, :, (seq_len//4):(seq_len//4)*2, :]
                    b= batch[feat][cropcat_idxs, :, (seq_len//4)*3:, :]
                    batch[feat][:n_cropcat, :, (seq_len//4):(seq_len//4)*2, :] = a
                    batch[feat][:n_cropcat, :, (seq_len//4)*3:, :] = b

        return batch, labels, total_votes, cropcat_idxs, half_seg

    
    def forward(self, x, cropcat_idxs=None, half_seg=None):
        if cropcat_idxs is None:
            return self.model(x)
        else:
            return self.model(x, cropcat_idxs, half_seg)
    
    def _shared_step(self, batch, stage, batch_idx):

        labels = batch.pop("label")
        label_ids = batch.pop("label_id").cpu()
        patient_ids = batch.pop("patient_id").cpu()
        total_votes = batch.pop("total_votes")
        
        # # Debug
        # for k,v in batch.items():
        #     print(k, v.shape)

        # CropCat Augmentation
        if stage == "train":
            batch, labels, total_votes, cropcat_idxs, half_seg = self.cropcat(batch, labels, total_votes)
        else:
            cropcat_idxs= None
            half_seg= None
        
        # Fwd Pass
        if self.cfg.encoder_model_type == "milAllbfg":
            y_logits, spec_logits, eeg_logits, eeg1d_logits = self(batch, cropcat_idxs, half_seg)
        else:
            y_logits = self(batch, cropcat_idxs, half_seg)

        if stage == "train":
            # Aux Heads
            if self.cfg.encoder_model_type == "milAllbfg":
                loss = self.loss_fn(y_logits, labels, total_votes)*0.79 + \
                       self.loss_fn(spec_logits, labels, total_votes)*0.06 + \
                       self.loss_fn(eeg_logits, labels, total_votes)*0.06 + \
                       self.loss_fn(eeg1d_logits, labels, total_votes)*0.09
            else:
                loss = self.loss_fn(y_logits, labels, total_votes, label_smoothing=self.cfg.label_smoothing)
        else:
            loss = self.loss_fn(y_logits, labels, total_votes, label_smoothing=0.0)
            
            # EMA or SWA
            if self.cfg.ema:
                ema_logits = self.ema_model.module(batch)
                ema_loss = self.loss_fn(ema_logits, labels, total_votes, label_smoothing=0.0)
                self._log(stage="ema_val", loss=ema_loss, batch_size=len(labels))

            if self.cfg.swa and self.trainer.state.fn == pl.trainer.states.TrainerFn.VALIDATING: # only runs on trainer.validate()
                swa_logits = self.swa_model.module(batch)
                swa_loss = self.loss_fn(swa_logits, labels, total_votes, label_smoothing=0.0)
                self._log(stage="swa_val", loss=swa_loss, batch_size=len(labels))

        # Metrics
        if stage == "val":
            self.val_outputs.append({
                "y_logits": y_logits,
                "y_labels": labels,
                "label_ids": label_ids,
                "patient_ids": patient_ids,
            })

        self._log(stage, loss, batch_size=len(labels))

        return loss
    
    def validation_step(self, batch, batch_idx) -> None:
        self._shared_step(batch, "val", batch_idx)
        return
    
    def training_step(self, batch, batch_idx):
        if self.cfg.ema:
            self.ema_model.update(self.model)
        if self.cfg.swa and self.trainer.max_epochs - self.current_epoch <= self.cfg.swa_epochs:
            self.swa_model.update(self.model)

        return self._shared_step(batch, "train", batch_idx)
    
    def _log(self, stage, loss, batch_size) -> None:
        self.log(f"{stage}_loss", loss, prog_bar=True, batch_size=batch_size, sync_dist=True)
        # if stage == "val":
        #     self.log_dict(self.metrics[f"{stage}_metrics"], prog_bar=True, batch_size=batch_size, sync_dist=True)
        return
    
    def on_train_start(self) -> None:
        """
        Logs cfg to wandb.
        """
        def is_json_serializable(obj):
            try:
                json.dumps(obj)
                return True
            except TypeError:
                return False

        def log_config(data, parent_key=''):
            for key, value in data.items():
                current_key = f"{parent_key}.{key}" if parent_key else key
                
                if isinstance(value, SimpleNamespace):
                    log_config(value.__dict__, current_key)
                else:
                    if is_json_serializable(value):
                        wandb.log({key: value}, commit=False)

        if not self.cfg.no_wandb and not self.cfg.fast_dev_run:
            log_config(self.cfg.__dict__)
        return
    
    def on_train_end(self) -> None:
        if self.cfg.swa:
            self.swa_model.merge_weights()

        if self.cfg.save_weights:
            save_fpath = "models/{}_{}_{}.pt".format(
                self.cfg.backbone,
                self.cfg.val_fold,
                self.cfg.seed,
                )
            torch.save(self.model.state_dict(), os.path.join(self.cfg.data_dir, save_fpath))

            if self.cfg.ema:
                ema_save_fpath = save_fpath.replace(str(self.cfg.seed), str(self.cfg.seed)+"_ema")
                torch.save(self.ema_model.module.state_dict(), os.path.join(self.cfg.data_dir, ema_save_fpath))

            if self.cfg.swa:
                swa_save_fpath = save_fpath.replace(str(self.cfg.seed), str(self.cfg.seed)+"_swa")
                torch.save(self.swa_model.module.state_dict(), os.path.join(self.cfg.data_dir, swa_save_fpath))
                
            print("Saved model weights. ")
        return
    
    def on_validation_epoch_end(self) -> None:
        # Save predictions
        if self.cfg.save_preds:
            # Convert preds to npy
            print("Saving predictions...")
            preds = np.concatenate([
                x["y_logits"].cpu().float().numpy() for x in self.val_outputs
                ])
            label_ids = np.concatenate([x["label_ids"] for x in self.val_outputs])

            # Save preds
            for pred, label_id in zip(preds, label_ids):
                spath = os.path.join(self.cfg.data_dir, "preds/{}/".format(label_id))
                if not os.path.exists(spath):
                    os.mkdir(spath)

                fname = os.path.join(spath, "pred_{}.npy".format(self.cfg.seed))
                np.save(fname, pred)
        
        self.val_outputs.clear()
        return