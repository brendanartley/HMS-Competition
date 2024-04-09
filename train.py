import argparse
from copy import copy, deepcopy
import sys
import os
import importlib

import numpy as np
import lightning.pytorch as pl

from src.utils.helpers import set_seed
from src.utils.callbacks import load_logger_and_callbacks

from src.modules.datamodule import CustomDataModule
from src.modules.trainmodule import CustomTrainModule


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-C", "--config", help="config filename", default="cfg_wavenet")
    parser.add_argument("-G", "--gpu_id", default="", help="GPU ID")
    parser_args, other_args = parser.parse_known_args(sys.argv)

    # Use all GPUs unless specified
    if parser_args.gpu_id != "":
        os.environ['CUDA_VISIBLE_DEVICES'] = str(parser_args.gpu_id)

    # Load CFG
    cfg = copy(importlib.import_module('src.configs.{}'.format(parser_args.config)).cfg)
    cfg.config_file = parser_args.config
    print("config ->", cfg.config_file)

    # Overwrite other arguments
    if len(other_args) > 1:
        other_args = {v.split("=")[0].lstrip("-"):v.split("=")[1] for v in other_args[1:]}

        for key in other_args:
            
            # Nested config
            if "." in key:
                keys = key.split(".")
                assert len(keys) == 2

                print(f'overwriting cfg.{keys[0]}.{keys[1]}: {cfg.__dict__[keys[0]].__dict__[keys[1]]} -> {other_args[key]}')
                cfg_type = type(cfg.__dict__[keys[0]].__dict__[keys[1]])
                if cfg_type == bool:
                    cfg.__dict__[keys[0]],__dict__[keys[1]] = other_args[key] == 'True'
                elif cfg_type == type(None):
                    cfg.__dict__[keys[0]].__dict__[keys[1]] = other_args[key]
                else:
                    cfg.__dict__[keys[0]].__dict__[keys[1]] = cfg_type(other_args[key])
                print(cfg.__dict__[keys[0]].__dict__[keys[1]])

            # Main config
            elif key in cfg.__dict__:
                print(f'overwriting cfg.{key}: {cfg.__dict__[key]} -> {other_args[key]}')
                cfg_type = type(cfg.__dict__[key])
                if cfg_type == bool:
                    cfg.__dict__[key] = other_args[key] == 'True'
                elif cfg_type == type(None):
                    cfg.__dict__[key] = other_args[key]
                else:
                    cfg.__dict__[key] = cfg_type(other_args[key])
                print(cfg.__dict__[key])
    
    # Set seed
    if cfg.seed < 0:
        cfg.seed = np.random.randint(1_000_000)
    print("seed", cfg.seed)
    set_seed(cfg.seed)
    
    return cfg

def train(cfg):

    # Limit CPU if doing debug run
    if cfg.fast_dev_run == True:
        cfg.num_workers = 1

    # Logger + Callabacks
    logger, callbacks = load_logger_and_callbacks(
        cfg=cfg,
        metrics = {
            "val_loss": "min", 
            "train_loss": "min",
            # "val_rmse": "min",
            },
    )

    # Load PL Modules
    data_module = CustomDataModule(cfg=cfg)
    train_module = CustomTrainModule(cfg=cfg)

    # Trainer Args: https://lightning.ai/docs/pytorch/stable/common/trainer.html#benchmark
    trainer = pl.Trainer(
        callbacks=callbacks,
        logger=logger,
        accelerator=cfg.accelerator,
        benchmark=cfg.benchmark, # set to True if input size does not change (increases speed)
        deterministic=False,
        fast_dev_run=cfg.fast_dev_run,
        max_epochs=cfg.epochs,
        num_sanity_val_steps=cfg.num_sanity_val_steps,
        # val_check_interval=cfg.val_check_interval,
        overfit_batches=cfg.overfit_batches,
        precision=cfg.precision,
        accumulate_grad_batches=cfg.accumulate_grad_batches,
        enable_checkpointing=cfg.enable_checkpointing,
        gradient_clip_val=cfg.gradient_clip_val,
        # strategy="deepspeed_stage_2",
        # strategy="deepspeed_stage_2_offload",
    )

    if not cfg.val_mode:
        trainer.fit(train_module, datamodule=data_module)
        trainer.validate(train_module, datamodule=data_module)
    else:
        trainer.validate(train_module, datamodule=data_module)
    return

def main(cfg):

    # Pre-train
    if cfg.pretrain and cfg.val_mode == False:
        pretrain_cfg= deepcopy(cfg)
        pretrain_cfg.metadata= "pretrain_50sec_nooverlap.csv"
        pretrain_cfg.save_weights= True
        pretrain_cfg.seed= 1001 # Seed for pretraining weights..
        pretrain_cfg.epochs= 2
        pretrain_cfg.lr= 1e-5
        pretrain_cfg.scheduler= "Constant"
        pretrain_cfg.ema= False
        pretrain_cfg.swa= False
        pretrain_cfg.no_wandb= True

        train(pretrain_cfg)
        
        cfg.pretrain= False
        cfg.weights_path= "./data/models/{}_{}_1001.pt".format(cfg.backbone, cfg.val_fold)

    # Train
    train(cfg)
    return

if __name__ == "__main__":
    cfg = parse_args()
    main(cfg)