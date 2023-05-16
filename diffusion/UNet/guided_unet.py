from abc import abstractmethod

import math

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from .base import TimestepEmbedSequential, Downsample, Upsample, ResBlock
from ..attention import AttentionBlock
from .diffusion_unet import DiffusionUNet

from ..nn import (
    conv_nd,
    timestep_embedding,
)


class GuidedUNet(DiffusionUNet):

    def _init_input_block(cls, cfg):
        ch = int(cfg.channel_mult[0] * cfg.model_channels)
        input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(conv_nd(cfg.dims, cfg.in_channels, ch, 3, padding=1))]
        )
        feature_size = ch
        input_block_chans = [ch]
        ds = 1
        for level, mult in enumerate(cfg.channel_mult):
            for _ in range(cfg.num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        cfg.time_embed_dim,
                        cfg.dropout,
                        out_channels=int(mult * cfg.model_channels),
                        dims=cfg.dims,
                        use_checkpoint=cfg.use_checkpoint,
                        use_scale_shift_norm=cfg.use_scale_shift_norm,
                    )
                ]
                ch = int(mult * cfg.model_channels)
                if ds in cfg.attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=cfg.use_checkpoint,
                            num_heads=cfg.num_heads,
                            num_head_channels=cfg.num_head_channels,
                            use_new_attention_order=cfg.use_new_attention_order,
                        )
                    )
                input_blocks.append(TimestepEmbedSequential(*layers))
                feature_size += ch
                input_block_chans.append(ch)
            if level != len(cfg.channel_mult) - 1:
                out_ch = ch
                input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            cfg.time_embed_dim,
                            cfg.dropout,
                            out_channels=out_ch,
                            dims=cfg.dims,
                            use_checkpoint=cfg.use_checkpoint,
                            use_scale_shift_norm=cfg.use_scale_shift_norm,
                            down=True,
                        )
                        if cfg.resblock_updown
                        else Downsample(
                            ch, cfg.conv_resample, dims=cfg.dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                feature_size += ch
                
        return input_blocks, ch, input_block_chans, ds, feature_size

    @classmethod
    def _init_output_block(cls, cfg, ch, input_block_chans, ds):
        output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(cfg.channel_mult))[::-1]:
            for i in range(cfg.num_res_blocks + 1):
                layers, ch = cls._base_output_block(cfg, mult, input_block_chans, ds)
                if level and i == cfg.num_res_blocks:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            cfg.time_embed_dim,
                            cfg.dropout,
                            out_channels=out_ch,
                            dims=cfg.dims,
                            use_checkpoint=cfg.use_checkpoint,
                            use_scale_shift_norm=cfg.use_scale_shift_norm,
                            up=True,
                        )
                        if cfg.resblock_updown
                        else Upsample(ch, cfg.conv_resample, dims=cfg.dims, out_channels=out_ch)
                    )
                    ds //= 2
                output_blocks.append(TimestepEmbedSequential(*layers))
                feature_size += ch
                return output_blocks, feature_size


    def forward(self, x, timesteps, y=None):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        emb = emb + self.label_emb(y)
        return self._forward_unet(x, emb)
  
    @classmethod
    def from_config(cls, cfg):
        model_cfg = DiffusionUNet.get_config(cfg)
        model_cfg['resblock_updown'] = cfg.DENOISE.UNET.resblock_updown
        model_cfg['use_new_attention_order'] = cfg.DENOISE.UNET.use_new_attention_order
        model_cfg['num_classes'] = cfg.DENOISE.UNET.num_classes
        model_cfg['num_head_channels'] = cfg.DENOISE.UNET.num_head_channels
        return model_cfg
        
 
    @classmethod
    def from_config(cls, cfg):
        model_cfg = cls.get_config(cfg)   
        time_embed = cls._init_time_embed(model_cfg)
        input_block, ch, input_block_chans, feature_size = cls._init_input_block(model_cfg)
        middle_block = cls._init_middle_block(model_cfg, ch, input_block_chans)
        model_cfg['feature_size'] = feature_size + model_cfg.ch
        output_block, feature_size = cls._init_output_block(model_cfg)
        out = cls._init_out(model_cfg, feature_size)
        
        return {
            'time_embed': time_embed,
            'label_emb': nn.Embedding(cfg.num_classes, cfg.time_embed_dim),
            'input_block': input_block,
            'middle_block': middle_block,
            'output_block': output_block,
            'out': out
            }
       