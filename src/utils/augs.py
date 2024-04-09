import random
from albumentations.core.transforms_interface import BasicTransform
from torch.nn import functional as F
from albumentations import Compose, random_utils
import torch
import numpy as np
import math
import typing

from albumentations import OneOf, MotionBlur, MedianBlur, CoarseDropout, Resize
    
class NumpyToTorch(BasicTransform):
    """
    Convert numpy array to torch tensor.
    
    Args:
        None

    Targets:
        image

    Image types:
        float32 (seq_len, n_features, 1)
    """

    def __init__(
        self,
        always_apply=False,
        p=1.0,
    ):
        super().__init__(always_apply, p)

    def apply(self, data, **params):
        return torch.from_numpy(data)
    
    @property
    def targets(self):
        return {"image": self.apply} 
    
class NormFillNAN(BasicTransform):
    """
    Normalize and fill NAN values.
    
    Args:
        None

    Targets:
        image

    Image types:
        float32 (seq_len, n_features, 1)
    """

    def __init__(
        self,
        always_apply=False,
        p=1.0,
    ):
        super().__init__(always_apply, p)

    def apply(self, data, **params):
        # Stats
        seq_mean = np.nanmean(data)
        seq_std = np.nanstd(data)+1e-6

        # Catches w/ ALL NAN
        if np.isnan(seq_mean) or np.isnan(seq_std) or seq_mean<1:
            data = np.nan_to_num(data, nan=0.0)
        else:
            data = np.nan_to_num(data, nan=seq_mean)

        # Norm
        data = (data - seq_mean) / seq_std
        data = np.nan_to_num(data, nan=0.0)

        # Clip large outliers
        data = np.clip(data, -16.0, 16.0)
        return data
    
    @property
    def targets(self):
        return {"image": self.apply} 

class TemporalResample(BasicTransform):
    """
    stretches/squeezes input over time dimension
    
    Args:
        sample_rate (float,float): lower and upper amount of resampling rate. Should both be float

    Targets:
        image

    Image types:
        float32 (seq_len, n_features, 1)
    """

    def __init__(
        self,
        sample_rate=(0.8,1.2),
        always_apply=False,
        p=0.5,
    ):
        super().__init__(always_apply, p)
        
        rate_lower = sample_rate[0]
        rate_upper = sample_rate[1]
        if not 0 <= rate_lower <= rate_upper:
            raise ValueError("Invalid combination of rate_lower and rate_upper. Got: {}".format((rate_lower, rate_upper)))

        self.rate_lower = rate_lower
        self.rate_upper = rate_upper

    def apply(self, data, **params):
        length = data.shape[0]
        new_size = max(int(length * params["sample_rate"]),1)
        new_x = F.interpolate(data.permute(2,1,0), new_size, mode="linear").permute(2,1,0)
        return new_x

    def get_params(self):
        return {"sample_rate": random.uniform(self.rate_lower, self.rate_upper)}

    def get_transform_init_args_names(self):
        return ("rate_lower", "rate_upper")
    
    @property
    def targets(self):
        return {"image": self.apply} 
    
class TemporalPad(BasicTransform):
    """
    Zero-pad the time dimension up to a specified size.
    
    Args:
        length (int): Pad up to this size

    Targets:
        image

    Image types:
        float32 (seq_len, n_features, 1)
    """

    def __init__(
        self,
        length=300,
        always_apply=False,
        p=1.0,
    ):
        super().__init__(always_apply, p)
        self.length = length

    def apply(self, data, **params):
        seq_len = data.shape[0]
        pad_len = max(self.length-seq_len, 0)
        if pad_len > 0:
            pad_left = pad_len//2
            pad_right = pad_len-pad_left
            data = F.pad(data, (0,0,0,0,pad_left,pad_right))
        return data
    
    @property
    def targets(self):
        return {"image": self.apply} 
    
class TemporalCenterCrop(BasicTransform):
    """
    Center crop time dimension to a specific size.
    
    Args:
        length (int): Pad up to this size

    Targets:
        image

    Image types:
        float32 (seq_len, n_features, 1)
    """

    def __init__(
        self,
        length=300,
        always_apply=False,
        p=1.0,
    ):
        super().__init__(always_apply, p)
        self.length = length

    def apply(self, data, **params):
        seq_len = data.shape[0]
        crop_len = max(seq_len - self.length, 0)
        if crop_len > 0:
            crop_left = crop_len//2
            crop_right = crop_len-crop_left
            data = data[crop_left:-crop_right, ...]
        return data
    
    @property
    def targets(self):
        return {"image": self.apply} 
    
    def get_transform_init_args_names(self):
        return ("length",)
    
class TemporalFlip(BasicTransform):
    """
    Flip the time dimension.
    
    Args:
        None

    Targets:
        image

    Image types:
        float32 (seq_len, n_features, 1)
    """

    def __init__(
        self,
        always_apply=False,
        p=1.0,
    ):
        super().__init__(always_apply, p)

    def apply(self, data, **params):
        return torch.flip(data, dims=[0])
    
    @property
    def targets(self):
        return {"image": self.apply} 
    
class FrequencyDropout(BasicTransform):
    """
    Random dropout of frequency channel.
    
    Args:
        None

    Targets:
        image

    Image types:
        float32 (seq_len, n_features, 1)
    """

    def __init__(
        self,
        always_apply=False,
        min_dropout= 0,
        max_dropout= 10,
        p=1.0,
    ):
        super().__init__(always_apply, p)
        self.min_dropout = min_dropout
        self.max_dropout = max_dropout

    def apply(self, data, **params):
        freq_size = data.shape[1]
        num_zeroed = torch.randint(self.min_dropout, self.max_dropout, size=(1,))
        if len(num_zeroed) != 0:
            zero_idxs = torch.randint(low=0, high=freq_size, size=(num_zeroed,))
            data = data.index_fill_(dim=1, index=zero_idxs, value=0.0)
        return torch.flip(data, dims=[0])
    
    @property
    def targets(self):
        return {"image": self.apply}

    def get_transform_init_args_names(self):
        return ("min_dropout", "max_dropout", ) 


class TemporalDropoutEdges(BasicTransform):
    """
    Dropout from the edges of the data.
    
    Args:
        sample_rate (float,float): lower and upper amount of dropout rate. Should both be float

    Targets:
        image

    Image types:
        float32 (seq_len, n_features, 1)
    """

    def __init__(
        self,
        sample_rate=(0.1,1.0),
        always_apply=False,
        p=0.5,
    ):
        super().__init__(always_apply, p)
        
        rate_lower = sample_rate[0]
        rate_upper = sample_rate[1]
        if not 0 <= rate_lower <= rate_upper:
            raise ValueError("Invalid combination of rate_lower and rate_upper. Got: {}".format((rate_lower, rate_upper)))

        self.rate_lower = rate_lower
        self.rate_upper = rate_upper

    def apply(self, data, **params):
        length = data.shape[0]
        mid= length // 2
        start_idx = int(mid - (params["sample_rate"]*mid))
        end_idx = int(mid + (params["sample_rate"]*mid))
        print(length, mid, start_idx, end_idx, params["sample_rate"])
        # new_size = max(int(length * params["sample_rate"]),1)
        # new_x = F.interpolate(data.permute(2,1,0), new_size, mode="linear").permute(2,1,0)
        return data

    def get_params(self):
        return {"sample_rate": random.uniform(self.rate_lower, self.rate_upper)}

    def get_transform_init_args_names(self):
        return ("rate_lower", "rate_upper")
    
    @property
    def targets(self):
        return {"image": self.apply} 
if __name__ == "__main__":

    transform = Compose([
        # -- Numpy Augs --
        # OneOf([
        #     MedianBlur(blur_limit=3, p=0.5),
        #     MotionBlur(blur_limit=3, p=0.5),
        # ], p=1.0),
        NormFillNAN(p=1.0),
        NumpyToTorch(p=1.0),
        # -- Torch Augs --
        TemporalDropoutEdges(sample_rate=(0.05, 1.0), p=1.0),
        # TemporalFlip(p=0.2),
        # TemporalResample(p=1.0, sample_rate=(0.8, 1.2)),
        # TemporalPad(p=1.0, length=300),
        # TemporalCenterCrop(p=1.0, length=300),
        FrequencyDropout(p=0.5, max_dropout=5),
    ])
    transform._disable_check_args() # disable type otherwise input must be numpy/ int8

    x = torch.rand(size=(300, 100, 4)).numpy()
    x = torch.rand(size=(300, 100, 4)).numpy()
    print(x.shape)

    z = transform(image=x)["image"]
    # z = z.numpy()
    print("out: ", z.shape)
    # print(z)


    
    