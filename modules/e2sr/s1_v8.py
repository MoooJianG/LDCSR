# 编码固定到 1/8
# 采用类似DiffIR的调制策略。

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.attention import LinearAttention
from .common import (
    ResnetBlock,
    Downsample,
    Normalize,
    nonlinearity,
    FRU,
    StyleLayer,
)
from modules.upsampler import SADNUpsampler


def get_scale_embedding(scale, embedding_dim, device=None):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    if not embedding_dim:
        return None

    scale = torch.Tensor([scale])
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=scale.device)
    emb = scale.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = F.pad(emb, (0, 1, 0, 0))
    return emb.to(device)


def make_coord(shape, ranges=None, flatten=True):
    """Make coordinates at grid centers."""
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=1, padding=1
            )

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class LinAttnBlock(LinearAttention):
    """Linear Attention Block"""

    def __init__(self, in_channels):
        super().__init__(dim=in_channels, heads=1, dim_head=in_channels)


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        q = q.reshape(b, c, h * w)
        q = q.permute(0, 2, 1)  # b,hw,c
        k = k.reshape(b, c, h * w)  # b,c,hw
        w_ = torch.bmm(q, k)  # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c) ** (-0.5))
        w_ = F.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b, c, h * w)
        w_ = w_.permute(0, 2, 1)  # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v, w_)  # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b, c, h, w)

        h_ = self.proj_out(h_)

        return x + h_


def make_attn(in_channels, attn_type="vanilla"):
    assert attn_type in ["vanilla", "linear", "none"], f"attn_type {attn_type} unknown"
    print(f"making attention of type '{attn_type}' with {in_channels} in_channels")
    if attn_type == "vanilla":
        return AttnBlock(in_channels)
    elif attn_type == "none":
        return nn.Identity(in_channels)
    else:
        return LinAttnBlock(in_channels)


class SAPA(nn.Module):
    """Scale aware pixel attention"""

    def __init__(self, nf):
        super().__init__()

        self.temb_ch = nf

        self.conv = nn.Sequential(
            nn.Conv2d(nf * 2, nf // 2, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf // 2, nf, 1),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, scale):

        scale_emb = get_scale_embedding(
            scale, embedding_dim=self.temb_ch, device=x.device
        )
        scale_emb = (
            scale_emb.unsqueeze_(2)
            .unsqueeze_(3)
            .expand([x.shape[0], scale_emb.shape[1], x.shape[2], x.shape[3]])
        )

        y = torch.cat([x, scale_emb], dim=1)
        y = self.conv(y)
        y = self.sigmoid(y)
        out = torch.mul(x, y)

        return out


class ImplicitRescaler(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, levels):
        super().__init__()

        self.levels = levels
        # residual block, interpolate, residual block
        self.conv_in = nn.Conv2d(in_channels, mid_channels, kernel_size=1)
        self.conv_mid = nn.Conv2d(mid_channels, mid_channels, kernel_size=1)
        self.conv_out = nn.Conv2d(mid_channels, out_channels, kernel_size=1)

        self.scale_aware_layer = nn.Sequential(
            *[nn.Linear(1, 64), nn.ReLU(), nn.Linear(64, levels), nn.Sigmoid()]
        )

    def forward(self, x_list, out_size):
        assert len(x_list) == self.levels

        device = x_list[0].device
        r = torch.tensor([x_list[0].shape[2] / out_size[0]], device=device)
        scale_w = self.scale_aware_layer(r.unsqueeze(0))[0]

        x_resize_list = []
        for l, x in enumerate(x_list):
            x_resize = F.interpolate(
                x, size=out_size, mode="bilinear", align_corners=False
            )
            x_resize *= scale_w[l]
            x_resize_list.append(x_resize)

        x = torch.cat(x_resize_list, dim=1)

        x = self.conv_in(x)
        x = nonlinearity(x)

        x = self.conv_mid(x)
        x = nonlinearity(x)

        x = self.conv_out(x)
        return x


class Encoder(nn.Module):
    # 编码到 1/8
    def __init__(
        self,
        *,
        ch,
        ch_mult=(1, 2, 4, 4),
        num_res_blocks,
        dropout=0.0,
        resamp_with_conv=True,
        in_channels=6,
        z_channels,
        double_z=True,
        **ignore_kwargs,
    ):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.in_channels = in_channels

        # downsampling
        self.conv_in = nn.Conv2d(
            in_channels, self.ch, kernel_size=3, stride=1, padding=1
        )

        in_ch_mult = (1,) + tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks):
                block.append(
                    ResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=self.temb_ch,
                        dropout=dropout,
                    )
                )
                block_in = block_out

            down = nn.Module()
            down.block = block
            down.attn = attn

            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, resamp_with_conv)
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )
        self.mid.block_2 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = nn.Conv2d(
            block_in,
            2 * z_channels if double_z else z_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(self, hr, lr):
        temb = None

        if hr is not None:
            lr_up = F.interpolate(
                lr, size=hr.shape[2:], mode="bicubic", align_corners=False
            )
            _input = torch.cat([lr_up, hr], dim=1)
        else:
            _input = lr

        # downsampling
        hs = [self.conv_in(_input)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)

            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.block_2(h, temb)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)

        return h


class MultiScaleDecoder(nn.Module):
    def __init__(
        self,
        *,
        ch,
        out_ch,
        ch_mult=(1, 2, 4, 8),
        num_res_blocks,
        attn_resolutions,
        dropout=0.0,
        resamp_with_conv=True,
        in_channels,
        resolution=256,
        z_channels,
        give_pre_end=False,
        tanh_out=False,
        attn_type="vanilla",
        num_sr_modules=[8, 8, 8, 8],
        **ignorekwargs,
    ):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.num_sr_modules = num_sr_modules
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end
        self.tanh_out = tanh_out

        assert len(num_sr_modules) == len(ch_mult)

        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,) + tuple(ch_mult)
        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1, z_channels, curr_res, curr_res)

        print(
            "Working with z of shape {} = {} dimensions.".format(
                self.z_shape, np.prod(self.z_shape)
            )
        )

        # z to block_in
        self.conv_in = nn.Conv2d(
            z_channels, block_in, kernel_size=3, stride=1, padding=1
        )

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks + 1):
                block.append(
                    ResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=self.temb_ch,
                        dropout=dropout,
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            up = nn.Module()
            up.block = block
            up.attn = attn

            # if i_level != 0:
            #     up.upsample = Upsample(block_in, resamp_with_conv)
            #     curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.implicit_rescaler = ImplicitRescaler(
            in_channels=block_in * np.array(ch_mult).sum(),
            mid_channels=256,
            out_channels=3,
            levels=self.num_resolutions,
        )

        ############# sr branch #############
        self.head = nn.Conv2d(in_channels, ch, 3, padding=1)

        self.FRUList = nn.ModuleList()
        self.StyleList = nn.ModuleList()
        for ch_mult, num in zip(reversed(ch_mult), self.num_sr_modules):
            self.StyleList.append(StyleLayer(ch, k_v_dim=ch * ch_mult))
            self.FRUList.append(FRU(ch, num_modules=num))

        self.upsampler = SADNUpsampler(ch, kSize=3, out_channels=out_ch)

    def get_last_layer(self):
        return self.upsampler.fuse[-1].weight

    def forward(self, z, lr, out_size):
        ############# decode branch #############
        temb = None  # get_scale_embedding(scale, embedding_dim=self.temb_ch, device=z.device)

        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        h_list = []
        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h, temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)

            # if i_level != 0:
            #     h = self.up[i_level].upsample(h)
            h_list.append(F.interpolate(h, size=lr.shape[2:], mode="nearest"))

        # end
        if self.give_pre_end:
            return h

        h_norm_list = []
        for l in range(self.num_resolutions):
            h_norm_list.append(nonlinearity(h_list[l]))

        ############# sr branch #############
        head = self.head(lr)
        mid = head
        for i_level in range(self.num_resolutions):
            mid = self.StyleList[i_level](mid, h_list[i_level])
            mid = self.FRUList[i_level](mid)

        x_up = F.interpolate(lr, out_size, mode="bicubic", align_corners=False)
        out = self.upsampler(mid, out_size) + x_up

        return out
