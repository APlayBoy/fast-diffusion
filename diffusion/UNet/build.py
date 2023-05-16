# encoding: utf-8
"""
@author:  xuxiaoqiang
@contact: www.xuxiaoqiang@gmail.com
"""

from ..utils.registry import Registry

DIFFUSION_UNET_REGISTRY = Registry("UNET")
DIFFUSION_UNET_REGISTRY.__doc__ = """
Registry for diffusion unet in models.
.
"""


def build_unet(cfg):
    """
    Build REIDHeads defined by `cfg.MODEL.REID_HEADS.NAME`.
    """
    unet = cfg.DENOISE.UNET.NAME
    return DIFFUSION_UNET_REGISTRY.get(unet)(cfg)