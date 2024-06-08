import copy
import math
from collections import OrderedDict
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch import nn
from torch.functional import Tensor
from torch.types import Device

from .constant import *
from .pde_base import PDE_NN_BASE, ConvBlock, LaplacianBlock

__all__ = ["FourierCNN"]


class FourierCNN(PDE_NN_BASE):
    def __init__(
        self,
        img_size: int = 256,
        in_channels: int = 1,
        out_channels: int = 2,
        in_frames: int = 8,
        offset_frames: int = 8,
        input_cfg: dict = {},
        guidance_generator_cfg: dict = {},
        encoder_cfg: dict = {},
        backbone_cfg: dict = {},
        decoder_cfg: dict = {},
        dropout_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        device: Device = torch.device("cuda:0"),
        aux_head: bool = False,
        aux_stide: List[int] = [2, 2, 2],
        aux_padding: List[int] = [1, 1, 1],
        aux_kernel_size_list: List[int] = [3, 3, 3],
        field_norm_mode: str = "max",
        num_iters: int = 1,
        eps_lap: bool = False,
        pac: bool = False,
        max_propagating_filter: int = 0,
        **kwargs,
    ):

        super().__init__(
            encoder_cfg=encoder_cfg,
            backbone_cfg=backbone_cfg,
            decoder_cfg=decoder_cfg,
            **kwargs,
        )

        """
        The overall network. It contains several conv layer.
        1. the total of the conv network should be about 16*16 since the FDTD is a local behavior
        2. TODO need to think about which makes more sense, bn or ln

        input: the solution of the coefficient function and locations (a(x, y), x, y)
        input shape: (batchsize, f = 1+8+50, x, y)
        output: the solution of next 50 frames (bs, f = 50, x ,y)
        """
        self.img_size = img_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_frames = in_frames
        self.offset_frames = offset_frames
        self.guidance_generator_cfg = guidance_generator_cfg
        self.encoder_cfg = encoder_cfg
        self.backbone_cfg = backbone_cfg
        self.decoder_cfg = decoder_cfg
        self.dropout_rate = dropout_rate
        self.drop_path_rate = drop_path_rate

        self.aux_head = aux_head
        self.aux_stide = aux_stide
        self.aux_padding = aux_padding
        self.aux_kernel_size_list = aux_kernel_size_list

        self.field_norm_mode = field_norm_mode
        self.num_iters = num_iters
        self.eps_lap = eps_lap
        self.pac = pac
        self.max_propagating_filter = max_propagating_filter

        self.input_cfg = input_cfg
        if self.input_cfg.input_mode == "eps_E0_Ji":
            self.in_channels = 1 + self.offset_frames + self.out_channels
            if not self.input_cfg.include_src:
                self.in_channels -= 1
        elif self.input_cfg.input_mode == "E0_Ji":
            self.in_channels = self.in_frames + self.out_channels
        elif self.input_cfg.input_mode == "eps_E0_lap_Ji":
            self.in_channels = 1 + self.offset_frames + 1 + self.out_channels
            if not self.input_cfg.include_src:
                self.in_channels -= 1

        self.device = device

        if self.backbone_cfg.share_weight:
            assert self.backbone_cfg.num_shared_layers > 1

        self.build_layers()

    def build_modules(self, name, cfg, in_channels):
        features = OrderedDict()
        for idx, out_channels in enumerate(cfg.kernel_list, 0):
            layer_name = "conv" + str(idx + 1)
            in_channels = in_channels if (idx == 0) else cfg.kernel_list[idx - 1]
            features[layer_name] = ConvBlock(
                in_channels,
                out_channels,
                cfg.kernel_size_list[idx],
                cfg.stride_list[idx],
                cfg.padding_list[idx],
                cfg.dilation_list[idx],
                cfg.groups_list[idx],
                bias=True,
                conv_cfg=cfg.conv_cfg,
                norm_cfg=(
                    cfg.norm_cfg
                    if cfg.norm_list[
                        idx
                    ]  # use norm_list to control the norm, more flexible
                    else None
                ),  # enable layer_norm
                act_cfg=(
                    cfg.act_cfg
                    if cfg.act_list[
                        idx
                    ]  # use act_list to control the act, more flexible
                    else None
                ),  # enable gelu
                residual=cfg.residual[idx],
                se=cfg.se[idx],
                pac=cfg.pac,
                with_cp=cfg.with_cp,
                device=self.device,
                if_pre_dwconv=cfg.if_pre_dwconv,
            )

        self.register_module(name, nn.Sequential(features))
        return getattr(self, name), out_channels

    def build_layers(self):
        if self.max_propagating_filter != 0:
            self.output_gate = nn.MaxPool2d(
                kernel_size=self.max_propagating_filter,
                stride=1,
                padding=self.max_propagating_filter // 2,
                )
        else:
            self.output_gate = None
        if self.pac:
            assert (
                self.input_cfg.input_mode == "E0_Ji"
            ), "when pac, input_mode must be E0_Ji"
            self.guidance_generator, guidance_channels = self.build_modules(
                "guidance_generator", self.guidance_generator_cfg, 1
            )

        self.encoder, out_channels = self.build_modules(
            "encoder", self.encoder_cfg, self.in_channels
        )
        if "3d" in self.backbone_cfg.conv_cfg["type"]:
            new_out_channels = 1
        else:
            new_out_channels = out_channels

        self.backbone, ret_out_channels = self.build_modules(
            "backbone", self.backbone_cfg, new_out_channels
        )
        if "3d" in self.backbone_cfg.conv_cfg["type"]:
            new_out_channels = out_channels
        else:
            new_out_channels = ret_out_channels

        if (
            self.decoder_cfg.kernel_list[-1] != self.out_channels
        ):  # make sure the last layer is the out_channels
            self.decoder_cfg.kernel_list[-1] = self.out_channels
        self.decoder, _ = self.build_modules(
            "decoder", self.decoder_cfg, new_out_channels
        )
        if "lap" in self.input_cfg.input_mode:
            self.laplacian = LaplacianBlock(device=self.device)

    def get_scaling_factor(self, input_fields, src):
        input_fields = input_fields.data
        if self.field_norm_mode == "max":
            abs_values = torch.abs(input_fields)
            scaling_factor = (
                abs_values.amax(dim=(1, 2, 3), keepdim=True) + 1e-6
            )  # bs, 1, 1, 1
        elif self.field_norm_mode == "avg_pool_max":
            abs_in_field = torch.abs(input_fields)
            _, idx = abs_in_field.mean(dim=(-1, -2)).max(dim=-1)
            abs_in_field = abs_in_field[:, idx] # bs, 1, h, w
            scaling_factor = torch.nn.functional.adaptive_avg_pool2d(abs_in_field, (abs_in_field.shape[-2]//8, abs_in_field.shape[-1]//8))
            scaling_factor = scaling_factor.amax(dim=(-1, -2), keepdim=True)

        elif self.field_norm_mode == "avg_pool_max99":
            abs_in_field = torch.abs(input_fields)
            _, idx = abs_in_field.mean(dim=(-1, -2)).max(dim=-1)
            abs_in_field = abs_in_field[:, idx] # bs, 1, h, w
            scaling_factor = torch.nn.functional.adaptive_avg_pool2d(abs_in_field, (abs_in_field.shape[-2]//8, abs_in_field.shape[-1]//8))
            scaling_factor = torch.quantile(scaling_factor.flatten(-2), 0.995, dim=-1, keepdim=True).unsqueeze(-1) # bs, 1, 1, 1

        elif self.field_norm_mode == "rigional_std":
            # the shape of the input field is [bs, channels, h, w] and it represents the E field along different time steps
            pixel_avg_energy = input_fields.square().mean(dim=-3, keepdim=True) # bs, 1, h, w
            max_avg_energy = pixel_avg_energy.amax(dim=(-2, -1), keepdim=True) # bs, 1, 1, 1
            regional_mask = pixel_avg_energy > max_avg_energy * 0.01 # bs, 1, h, w
            regional_mask = regional_mask.expand(-1, input_fields.size(1), -1, -1)
            scaling_factor = torch.stack([input_fields[b][regional_mask[b]].std() for b in range(input_fields.size(0))]).view(-1, 1, 1, 1)
            

        elif self.field_norm_mode == "max99":
            scaling_factor = torch.quantile(input_fields.abs().flatten(-3), q=0.9995, dim=-1, keepdim=True).view(-1, 1, 1, 1)
        elif self.field_norm_mode == "std":
            scaling_factor = 15 * input_fields.std(dim=(1, 2, 3), keepdim=True)
        elif self.field_norm_mode == "none":  # we don't scale the input fields anymore
            scaling_factor = torch.ones(
                (input_fields.size(0), 1, 1, 1), device=input_fields.device
            )
        elif self.field_norm_mode == "max_w_src":
            E0_src = torch.cat([input_fields, src], dim=1)
            abs_values = torch.abs(E0_src)
            scaling_factor = (
                abs_values.amax(dim=(1, 2, 3), keepdim=True) + 1e-6
            )  # the smallest Emax value is of magnitude 1e-6, so the eps added should be smaller than that
            # bs, 1, 1, 1
        else:
            raise NotImplementedError

        return scaling_factor  # [bs, 1, 1, 1]
    
    def get_propagating_mask(self, input_fields, src):
        field_energy = input_fields.data.square().mean(dim = 1, keepdim=True)
        src_energy = src.data.square().mean(dim = 1, keepdim=True)
        avg_energy = field_energy + src_energy
        propagating_mask = self.output_gate(avg_energy) # [bs, 1, h, w]
        propagating_mask = propagating_mask > torch.quantile(propagating_mask, 0.1)
        return propagating_mask


    def preprocess(self, x: Tensor) -> Tuple[Tensor, ...]:
        ## obtain the input fields and the source fields from input data
        eps = 1 / x[:, 0:1].square()
        input_fields = x[:, 1 : 1 + self.in_frames]
        srcs = x[:, 1 + self.in_frames :, ...].chunk(self.num_iters, dim=1)
        return eps, input_fields, srcs

    def forward(
        self,
        x,
        src_mask: Optional[Tensor] = None,
        padding_mask: Optional[Tensor] = None,
    ):
        eps, input_fields, srcs = self.preprocess(x)
        normalization_factor = []
        outs = []
        for iter, src in enumerate(srcs):
            scaling_factor = self.get_scaling_factor(input_fields, src)
            if self.max_propagating_filter != 0:
                propagating_mask = self.get_propagating_mask(input_fields, src)
            if self.input_cfg.input_mode == "eps_E0_Ji":
                x = torch.cat(
                    [eps, input_fields / scaling_factor, src / scaling_factor], dim=1
                )
                if not self.input_cfg.include_src:
                    x = x[:, :-1, ...]  # drop the last src
            elif self.input_cfg.input_mode == "E0_Ji":
                assert self.pac, "PAC must be true"
                assert (
                    self.guidance_generator is not None
                ), "guidance_generator must be provided"
                guidance, _ = self.guidance_generator((eps, None))
                x = torch.cat([input_fields, src], dim=1)
                if self.field_norm_mode != "none":
                    x.div_(scaling_factor)
            elif self.input_cfg.input_mode == "eps_E0_lap_Ji":
                divergence = self.laplacian(input_fields[:, -1:, ...] / scaling_factor)
                if self.eps_lap:
                    divergence = divergence * eps
                x = torch.cat(
                    [
                        eps,
                        input_fields / scaling_factor,
                        divergence,
                        src / scaling_factor,
                    ],
                    dim=1,
                )
                if not self.input_cfg.include_src:
                    x = x[:, :-1, ...]  # drop the last src
            x, _ = self.encoder((x, None))
            if self.backbone_cfg.pac:
                if self.backbone_cfg.share_weight:
                    for _ in range(self.backbone_cfg.num_shared_layers):
                        x, _ = self.backbone((x, guidance))
                else:
                    x, _ = self.backbone((x, guidance))
            else:
                if self.backbone_cfg.share_weight:
                    for _ in range(self.backbone_cfg.num_shared_layers):
                        x, _ = self.backbone((x, None))
                else:
                    x, _ = self.backbone((x, None))
            if self.decoder_cfg.pac:
                x, _ = self.decoder((x, guidance))
            else:
                x, _ = self.decoder((x, None))
            src_mask = src_mask >= 0.5
            src = src[:, -self.out_channels :, ...]
            if self.field_norm_mode != "none":
                src = src / scaling_factor
            out = torch.where(src_mask, src, x)
            out = out * padding_mask
            if self.max_propagating_filter != 0:
                out = out * propagating_mask

            normalization_factor.append(scaling_factor)
            outs.append(out)
            ## update input fields for next iteration
            input_fields = out[:, -self.in_frames :, ...]
            if self.field_norm_mode != "none":
                input_fields = input_fields * scaling_factor

        outs = torch.cat(outs, dim=1)

        normalization_factor = (
            torch.stack(normalization_factor, 1)
            .expand(-1, -1, self.out_channels, -1, -1)
            .flatten(1, 2)
        )

        return outs, normalization_factor
