import torch as th
from torch import nn
from .base import TimestepEmbedSequential, Downsample, Upsample, ResBlock
from ..attention import AttentionBlock

from ..nn import (
    conv_nd,
    linear,
    timestep_embedding,
    zero_module,
    normalization,
)


class UNetModel(nn.Module):
    """
    The full UNet model with attention and timestep embedding.
    """

    def __init__(
        self,
        *,
        time_embed,
        input_block,
        middle_block,
        output_block,
        out
       
    ):
        super().__init__()
        self.time_embed = time_embed
        self.input_block = input_block
        self.middle_block = middle_block
        self.output_block = output_block
        self.out = out
    
    @classmethod
    def _init_time_embed(cls, unet_cfg):
        return nn.Sequential(
            linear(unet_cfg.model_channels, unet_cfg.time_embed_dim),
            nn.siLU(),
            linear(unet_cfg.time_embed_dim, unet_cfg.time_embed_dim),
        )
    
    @classmethod
    def _init_input_block(cls, unet_cfg):
        input_blocks = nn.ModuleList(
            [
                conv_nd(unet_cfg.dims, unet_cfg.in_channels,unet_cfg. model_channels, 3, padding=1)
                
            ]
        )
        input_block_chans = [unet_cfg.model_channels]
        ch = unet_cfg.model_channels
        ds = 1
        for level, mult in enumerate(unet_cfg.channel_mult):
            for _ in range(unet_cfg.num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        unet_cfg.time_embed_dim,
                        unet_cfg.dropout,
                        out_channels=mult * unet_cfg.model_channels,
                        dims=unet_cfg.dims,
                        use_checkpoint=unet_cfg.use_checkpoint,
                        use_scale_shift_norm=unet_cfg.use_scale_shift_norm,
                    )
                ]
                ch = mult * unet_cfg.model_channels
                if ds in unet_cfg.attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch, use_checkpoint=unet_cfg.use_checkpoint, num_heads=unet_cfg.num_heads
                        )
                    )
                input_blocks.append(TimestepEmbedSequential(*layers))
                input_block_chans.append(ch)
            if level != len(unet_cfg.channel_mult) - 1:
                input_blocks.append(
                    TimestepEmbedSequential(Downsample(ch, unet_cfg.conv_resample, dims=unet_cfg.dims))
                )
                input_block_chans.append(ch)
                ds *= 2
        return input_blocks, ch, input_block_chans
        
    @classmethod
    def _init_middle_block(self, unet_cfg):
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                unet_cfg.ch,
                unet_cfg.time_embed_dim,
                unet_cfg.dropout,
                dims=unet_cfg.dims,
                use_checkpoint=unet_cfg.use_checkpoint,
                use_scale_shift_norm=unet_cfg.use_scale_shift_norm,
            ),
            AttentionBlock(unet_cfg.ch, use_checkpoint=unet_cfg.use_checkpoint, num_heads=unet_cfg.num_heads),
            ResBlock(
                unet_cfg.ch,
                unet_cfg.time_embed_dim,
                unet_cfg.dropout,
                dims=unet_cfg.dims,
                use_checkpoint=unet_cfg.use_checkpoint,
                use_scale_shift_norm=unet_cfg.use_scale_shift_norm,
            ),
        )
        
    @classmethod
    def _init_output_block(self, cfg):
        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(cfg.channel_mult))[::-1]:
            for i in range(cfg.num_res_blocks + 1):
                layers = [
                    ResBlock(
                        ch + cfg.input_block_chans.pop(),
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
                if level and i == cfg.num_res_blocks:
                    layers.append(Upsample(ch, cfg.conv_resample, dims=cfg.dims))
                    ds //= 2
                self.output_blocks.append(nn.Sequential(*layers))
       
    @classmethod
    def _init_out(self, cfg):
        self.out = nn.Sequential(
            normalization(cfg.ch),
            nn.SiLU(),
            zero_module(conv_nd(cfg.dims, cfg.model_channels,cfg. out_channels, 3, padding=1)),
        )


    @property
    def inner_dtype(self):
        """
        Get the dtype used by the torso of the model.
        """
        return next(self.input_blocks.parameters()).dtype

    # def forward(self, x):
        
    def forward(self, x, timesteps,):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """

        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))


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
        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

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

    @classmethod
    def from_config(cls, cfg):
        
        in_channel = cfg.DENOISE.UNET.in_channel,
        model_channels = cfg.DENOISE.UNET.model_channels,
        out_channels = cfg.DENOISE.UNET.out_channels,
        num_res_blocks = cfg.DENOISE.UNET.num_res_blocks,
        attention_resolutions = cfg.DENOISE.UNET.attention_resolutions,
        dropout = cfg.DENOISE.UNET.dropout,
        channel_mult = cfg.DENOISE.UNET.channel_mult,
        conv_resample= cfg.DENOISE.UNET.conv_resample,
        dims = cfg.DENOISE.UNET.dims,
        use_checkpoint= cfg.DENOISE.UNET.use_checkpoint,
        num_heads= cfg.DENOISE.UNET.num_heads,
        if num_heads_upsample == -1:
            num_heads_upsample = num_heads
        else:
            num_heads_upsample=cfg.DENOISE.UNET.num_heads_upsample,
        use_scale_shift_norm=cfg.DENOISE.UNET.use_scale_shift_norm
        unet_cfg = {
            'in_channel':in_channel,
            'model_channels':model_channels,
            'out_channels':out_channels,
            'num_res_blocks':num_res_blocks,
            'attention_resolutions':attention_resolutions,
            'dropout':dropout,
            'channel_mult':channel_mult,
            'conv_resample':conv_resample,
            'dims':dims,
            'use_checkpoint':use_checkpoint,
            'num_heads':num_heads,
            'num_heads_upsample':num_heads_upsample,
            'use_scale_shift_norm':use_scale_shift_norm,
            'time_embed_dim': model_channels * 4
        }
        time_embed = cls._init_time_embed(unet_cfg)
        input_block, ch, input_block_chans = cls._init_input_block(unet_cfg)
        unet_cfg['ch'] = ch
        unet_cfg['input_block_chans'] = input_block_chans
        middle_block = cls._init_middle_block(unet_cfg)
        output_block = cls._init_output_block(unet_cfg)
        out = cls._init_out(unet_cfg)
        
        return {
            'time_embed': time_embed,
            'input_block': input_block,
            'middle_block': middle_block,
            'output_block': output_block,
            'out': out
            }
       
       
       
       