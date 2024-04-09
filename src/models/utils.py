import torch
import torch.nn as nn

def count_parameters(model):
    return "{:_}".format(sum(p.numel() for p in model.parameters() if p.requires_grad))