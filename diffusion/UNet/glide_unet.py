import math

import torch as th
import torch.nn as nn
import torch.nn.functional as F
from .guided_unet import GuidedUNet


class SuperResUNetModel(GuidedUNet):
        
    @classmethod
    def get_config(cls, cfg):
        model_cfg = GuidedUNet.get_config(cfg)
        model_cfg['in_channels'] = model_cfg['in_channels'] * 2
        return model_cfg
        

    def forward(self, x, timesteps, low_res=None, **kwargs):
        _, _, new_height, new_width = x.shape
        upsampled = F.interpolate(low_res, (new_height, new_width), mode="bilinear")
        x = th.cat([x, upsampled], dim=1)
        return super().forward(x, timesteps, **kwargs)

    
class InpaintUNetModel(GuidedUNet):
    """
    A UNetModel which can perform inpainting.
    """

    @classmethod
    def get_config(cls, cfg):
        model_cfg = GuidedUNet.get_config(cfg)
        model_cfg['in_channels'] = model_cfg['in_channels'] * 2 + 1
        return model_cfg

    def forward(self, x, timesteps, inpaint_image=None, inpaint_mask=None, **kwargs):
        if inpaint_image is None:
            inpaint_image = th.zeros_like(x)
        if inpaint_mask is None:
            inpaint_mask = th.zeros_like(x[:, :1])
        return super().forward(
            th.cat([x, inpaint_image * inpaint_mask, inpaint_mask], dim=1),
            timesteps,
            **kwargs,
        )


class SuperResInpaintUNetModel(GuidedUNet):
    """
    A UNetModel which can perform both upsampling and inpainting.
    """

    @classmethod
    def get_config(cls, cfg):
        model_cfg = GuidedUNet.get_config(cfg)
        model_cfg['in_channels'] = model_cfg['in_channels'] * 3  + 1
        return model_cfg

    def forward(
        self,
        x,
        timesteps,
        inpaint_image=None,
        inpaint_mask=None,
        low_res=None,
        **kwargs,
    ):
        if inpaint_image is None:
            inpaint_image = th.zeros_like(x)
        if inpaint_mask is None:
            inpaint_mask = th.zeros_like(x[:, :1])
        _, _, new_height, new_width = x.shape
        upsampled = F.interpolate(low_res, (new_height, new_width), mode="bilinear")
        return super().forward(
            th.cat([x, inpaint_image * inpaint_mask, inpaint_mask, upsampled], dim=1),
            timesteps,
            **kwargs,
        )