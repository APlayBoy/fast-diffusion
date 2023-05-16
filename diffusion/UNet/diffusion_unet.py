import torch as th
from torch import nn
from .base import TimestepEmbedSequential, Downsample, Upsample, ResBlock
from ..attention import AttentionBlock
from .build import DIFFUSION_UNET_REGISTRY
from config import configurable

from ..nn import (
    conv_nd,
    linear,
    timestep_embedding,
    zero_module,
    normalization,
)


@DIFFUSION_UNET_REGISTRY.register()
class DiffusionUNet(nn.Module):
    """
    The full UNet model with attention and timestep embedding.
    """

    @configurable
    def __init__(
        self,
        *,
        time_embed,
        label_emb,
        input_block,
        middle_block,
        output_block,
        out
       
    ):
        super().__init__()
        self.time_embed = time_embed
        self.label_emb = label_emb
        self.input_block = input_block
        self.middle_block = middle_block
        self.output_block = output_block
        self.out = out
    
    @classmethod
    def _init_time_embed(cls, cfg):
        return nn.Sequential(
            linear(cfg.model_channels, cfg.time_embed_dim),
            nn.siLU(),
            linear(cfg.time_embed_dim, cfg.time_embed_dim),
        )
    
    @classmethod
    def _init_input_block(cls, cfg):
        input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(cfg.dims, cfg.in_channels,cfg.model_channels, 3, padding=1)
                )
            ]
        )
        ch = int(cfg.channel_mult[0] * cfg.model_channels)
        input_block_chans = [ch]
        ds = 1
        for level, mult in enumerate(cfg.channel_mult):
            for _ in range(cfg.num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        cfg.time_embed_dim,
                        cfg.dropout,
                        out_channels=mult * cfg.model_channels,
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
                            num_heads=cfg.num_heads
                        )
                    )
                input_blocks.append(TimestepEmbedSequential(*layers))
                input_block_chans.append(ch)
            if level != len(cfg.channel_mult) - 1:
                input_blocks.append(
                    TimestepEmbedSequential(Downsample(ch, cfg.conv_resample, dims=cfg.dims))
                )
                input_block_chans.append(ch)
                ds *= 2
        return input_blocks, ch, input_block_chans, ds
        
    @classmethod
    def _init_middle_block(self, cfg, ch):
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                cfg.time_embed_dim,
                cfg.dropout,
                dims=cfg.dims,
                use_checkpoint=cfg.use_checkpoint,
                use_scale_shift_norm=cfg.use_scale_shift_norm,
            ),
            AttentionBlock(ch, 
                           use_checkpoint=cfg.use_checkpoint, 
                           num_heads=cfg.num_heads),
            ResBlock(
                ch,
                cfg.time_embed_dim,
                cfg.dropout,
                dims=cfg.dims,
                use_checkpoint=cfg.use_checkpoint,
                use_scale_shift_norm=cfg.use_scale_shift_norm,
            ),
        )
    
    def _base_output_block(cls, cfg, mult, input_block_chans, ds):
        layers = [
            ResBlock(
                ch + input_block_chans.pop(),
                cfg.time_embed_dim,
                cfg.dropout,
                out_channels=cfg.model_channels * mult,
                dims=cfg.dims,
                use_checkpoint=cfg.use_checkpoint,
                use_scale_shift_norm=cfg.use_scale_shift_norm,
            )
        ]
        ch = cfg.model_channels * mult
        if ds in cfg.attention_resolutions:
            layers.append(
                AttentionBlock(
                    ch,
                    use_checkpoint=cfg.use_checkpoint,
                    num_heads=cfg.num_heads_upsample,
                )
            )
        return layers, ch
    
        
    @classmethod
    def _init_output_block(cls, cfg, ch, input_block_chans, ds):
        output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(cfg.channel_mult))[::-1]:
            for i in range(cfg.num_res_blocks + 1):        
                layers, ch = cls._base_output_block(cfg, mult, input_block_chans, ds)
                if level and i == cfg.num_res_blocks:
                    layers.append(Upsample(ch, cfg.conv_resample, dims=cfg.dims))
                    ds //= 2
                output_blocks.append(TimestepEmbedSequential(*layers))
        return output_blocks
       
    @classmethod
    def _init_out(self, cfg):
        self.out = nn.Sequential(
            normalization(cfg.ch),
            nn.SiLU(),
            zero_module(conv_nd(cfg.dims, cfg.model_channels, cfg.out_channels, 3, padding=1)),
        )

    @property
    def inner_dtype(self):
        """
        Get the dtype used by the torso of the model.
        """
        return next(self.input_blocks.parameters()).dtype

    def _forward_unet(self, x, emb=None):
        hs = []
        h = x.type(self.inner_dtype)
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
        h = self.middle_block(h, emb)
        for module in self.output_blocks:
            cat_in = th.cat([h, hs.pop()], dim=1)
            h = module(cat_in, emb)
        h = h.type(x.dtype)
        return self.out(h)
        
    def forward(self, x, timesteps,):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        return self._forward_unet(x, emb)
    
    def _get_unet_vectors(self, x, emb):
        hs = []
        result = dict(down=[], up=[])
        h = x.type(self.inner_dtype)
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
            result["down"].append(h.type(x.dtype))
        h = self.middle_block(h, emb)
        result["middle"] = h.type(x.dtype)
        for module in self.output_blocks:
            cat_in = th.cat([h, hs.pop()], dim=1)
            h = module(cat_in, emb)
            result["up"].append(h.type(x.dtype))
        return result

    def get_feature_vectors(self, x, timesteps, y=None):
        """
        Apply the model and return all of the intermediate tensors.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: a dict with the following keys:
                 - 'down': a list of hidden state tensors from downsampling.
                 - 'middle': the tensor of the output of the lowest-resolution
                             block in the model.
                 - 'up': a list of hidden state tensors from upsampling.
        """
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        return self._get_unet_vectors(x, emb)
    
    @classmethod
    def get_config(cls, cfg):
        num_heads_upsample = cfg.DENOISE.UNET.num_heads_upsample
        num_heads = cfg.DENOISE.UNET.num_heads
        input_ch = int(cfg.DENOISE.UNET.channel_mult[0] * cfg.DENOISE.UNET.model_channels)
        return {
            'in_channel':cfg.DENOISE.UNET.in_channel,
            'model_channels':cfg.DENOISE.UNET.model_channels,
            'input_ch': input_ch,
            'out_channels':cfg.DENOISE.UNET.out_channels,
            'num_res_blocks':cfg.DENOISE.UNET.num_res_blocks,
            'attention_resolutions':cfg.DENOISE.UNET.attention_resolutions,
            'dropout':cfg.DENOISE.UNET.dropout,
            'channel_mult':cfg.DENOISE.UNET.channel_mult,
            'conv_resample':cfg.DENOISE.UNET.conv_resample,
            'dims':cfg.DENOISE.UNET.dims,
            'use_checkpoint':cfg.DENOISE.UNET.use_checkpoint,
            'num_heads':num_heads,
            'num_heads_upsample':num_heads if num_heads_upsample == -1 else num_heads_upsample,
            'use_scale_shift_norm':cfg.DENOISE.UNET.use_scale_shift_norm,
            'time_embed_dim': cfg.DENOISE.UNET.model_channels * 4
        }
        

    @classmethod
    def from_config(cls, cfg):
        model_cfg = cls.get_config(cfg)   
        time_embed = cls._init_time_embed(model_cfg)
        input_block, ch, input_block_chans, ds = cls._init_input_block(model_cfg)
        middle_block = cls._init_middle_block(model_cfg, ch)
        output_block = cls._init_output_block(model_cfg, ch, input_block_chans, ds)
        out = cls._init_out(model_cfg)
        
        return {
            'time_embed': time_embed,
            'label_emb': None,
            'input_block': input_block,
            'middle_block': middle_block,
            'output_block': output_block,
            'out': out
            }
       