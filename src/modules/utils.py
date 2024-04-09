from copy import deepcopy

import torch
import torch.nn as nn

class ModelEMA(nn.Module):
    """
    EMA for model weights.
    Source: https://www.kaggle.com/competitions/blood-vessel-segmentation/discussion/475080#2641635
    
    Ex.
    def training_step(self, batch, batch_idx):
        self.ema_model.update(self.model)
        ...
    """
    def __init__(self, model, decay=0.9999, device=None):
        super().__init__()
        # make a copy of the model for accumulating moving average of weights
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.device = device  # perform ema on different device from model if set
        if self.device is not None:
            self.module.to(device=device)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)

class ModelSWA(nn.Module):
    """
    Stochastic Weight Averaging.

    Ex.
    def on_train_end(self) -> None:
        self.swa_model.merge_weights()
        ...

    def training_step(self, batch, batch_idx):
        self.swa_model.update(self.model)
        ...
    """
    def __init__(self, model, device=None, epochs=1):
        super().__init__()
        self.step_count = -1
        self.device = device  # perform swa on different device from model if set
        self.module = deepcopy(model) # make a copy of the model for accumulating weights
        self.module.eval()
        self._update(model, update_fn=lambda s,m: s.zero_()) # init weights with 0s
        if self.device is not None:
            self.module.to(device=device)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for swa_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                swa_v.copy_(update_fn(swa_v, model_v))
        self.step_count += 1

    def update(self, model):
        self._update(model, update_fn=lambda s, m: s+m)

    def merge_weights(self,):
        print("SWA: merging weights..")
        with torch.no_grad():
            for swa_v in self.module.state_dict().values():
                if self.device is not None:
                    swa_v = swa_v.to(device=self.device)
                swa_v.copy_(swa_v / self.step_count)