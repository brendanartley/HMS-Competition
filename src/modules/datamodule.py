
import lightning.pytorch as pl
import torch
import numpy as np

from src.modules.dataset import CustomDataset

class CustomDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
    
    def setup(self, stage):        
        if stage == "fit" or stage is None:
            self.train_dataset = CustomDataset(cfg=self.cfg, mode="train")
            self.val_dataset = CustomDataset(cfg=self.cfg, mode="val")
            print("Dataset sizes. T: {:_}, V: {:_}".format(len(self.train_dataset), len(self.val_dataset)))

        elif stage == "validate":
            self.val_dataset = CustomDataset(cfg=self.cfg, mode="val")
    
    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size = self.cfg.batch_size,
            num_workers = self.cfg.num_workers,
            pin_memory = self.cfg.pin_memory,
            drop_last = True,
            shuffle = True,
        )
    
    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size = self.cfg.batch_size,
            num_workers = self.cfg.num_workers,
            pin_memory = self.cfg.pin_memory,
            drop_last = False,
            shuffle = False,
        )