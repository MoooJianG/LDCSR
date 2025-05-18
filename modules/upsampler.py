import math
import torch
import torch.nn as nn
import torch.nn.functional as F

################
# Upsampler
################


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


class PixShuffleUpsampler(nn.Sequential):
    def __init__(self, scale, n_feats, bn=False, act=False):
        m = []
        if (scale & (scale - 1)) == 0:  # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(n_feats, 4 * n_feats, kernel_size=3, padding=1))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == "relu":
                    m.append(nn.ReLU(True))
                elif act == "prelu":
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(nn.Conv2d(n_feats, 9 * n_feats, kernel_size=3, padding=1))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == "relu":
                m.append(nn.ReLU(True))
            elif act == "prelu":
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError
        m.append(nn.PReLU(n_feats))
        m.append(nn.Conv2d(n_feats, 3, 3, padding=1))

        super().__init__(*m)


class PA(nn.Module):
    """PA is pixel attention"""

    def __init__(self, nf):

        super(PA, self).__init__()
        self.conv = nn.Conv2d(nf, nf, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        y = self.conv(x)
        y = self.sigmoid(y)
        out = torch.mul(x, y)

        return out


class ScaleEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.dim = dim
        half_dim = self.dim // 2
        self.inv_freq = torch.exp(
            torch.arange(half_dim, dtype=torch.float32)
            * (-math.log(10000) / (half_dim - 1))
        )

    def forward(self, input):
        shape = input.shape
        input = input.view(-1).to(torch.float32)
        sinusoid_in = torch.ger(input, self.inv_freq.to(input.device))
        pos_emb = torch.cat([sinusoid_in.sin(), sinusoid_in.cos()], dim=-1)
        pos_emb = pos_emb.view(*shape, self.dim)

        return pos_emb


class SAPA(nn.Module):
    """Scale aware pixel attention"""

    def __init__(self, nf):
        super().__init__()

        self.scale_embing = ScaleEmbedding(nf)

        self.conv = nn.Sequential(
            nn.Conv2d(nf * 2, nf // 2, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf // 2, nf, 1),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, scale):

        scale_emb = self.scale_embing(scale)
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


class SalDRNUpsampler(nn.Module):
    # SalDRN Upsampler
    def __init__(self, n_feat, split=4):
        # final
        super().__init__()

        self.distilled_channels = n_feat // split
        self.out = nn.Sequential(
            nn.Conv2d(n_feat, n_feat // 4, 3, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(n_feat // 4, 3, 3, padding=1),
        )

        up = []
        up.append(
            nn.Conv2d(n_feat // split * 3, (n_feat // split) * 4 * 3, 3, 1, 1, groups=3)
        )
        up.append(nn.PixelShuffle(2))
        self.upsample = nn.Sequential(*up)

        up1 = []
        up1.append(
            nn.Conv2d(n_feat // split * 2, (n_feat // split) * 4 * 2, 3, 1, 1, groups=2)
        )
        up1.append(nn.PixelShuffle(2))
        self.upsample1 = nn.Sequential(*up1)

        up2 = []
        up2.append(nn.Conv2d(n_feat // split, (n_feat // split) * 4, 3, 1, 1))
        up2.append(nn.PixelShuffle(2))
        self.upsample2 = nn.Sequential(*up2)

        self.SAPA = SAPA(n_feat)

    def forward(self, x, out_size):
        scale = torch.tensor([x.shape[2] / out_size[0]], device=x.device)

        out1, remaining_c1 = torch.split(
            x, (self.distilled_channels, self.distilled_channels * 3), dim=1
        )
        out = self.upsample(remaining_c1)

        out2, remaining_c2 = torch.split(
            out, (self.distilled_channels, self.distilled_channels * 2), dim=1
        )
        out = self.upsample1(remaining_c2)

        out3, remaining_c3 = torch.split(
            out, (self.distilled_channels, self.distilled_channels), dim=1
        )
        out = self.upsample2(remaining_c3)

        distilled_c1 = F.interpolate(
            out1, out_size, mode="bilinear", align_corners=False
        )
        distilled_c2 = F.interpolate(
            out2, out_size, mode="bilinear", align_corners=False
        )
        distilled_c3 = F.interpolate(
            out3, out_size, mode="bilinear", align_corners=False
        )
        distilled_c4 = F.interpolate(
            out, out_size, mode="bilinear", align_corners=False
        )

        out = torch.cat([distilled_c1, distilled_c2, distilled_c3, distilled_c4], dim=1)

        out = self.out(self.SAPA(out, scale))
        return out


class SADNUpsampler(nn.Module):
    # Up-sampling of SADN
    def __init__(
        self, n_feats, kSize, out_channels, interpolate_mode="bilinear", levels=4
    ):
        super().__init__()
        self.interpolate_mode = interpolate_mode
        self.levels = levels

        self.UPNet_x2_list = []

        for _ in range(levels - 1):
            self.UPNet_x2_list.append(
                nn.Sequential(
                    *[
                        nn.Conv2d(
                            n_feats,
                            n_feats * 4,
                            kSize,
                            padding=(kSize - 1) // 2,
                            stride=1,
                        ),
                        nn.PixelShuffle(2),
                    ]
                )
            )

        self.scale_aware_layer = nn.Sequential(
            *[nn.Linear(1, 64), nn.ReLU(), nn.Linear(64, levels), nn.Sigmoid()]
        )

        self.UPNet_x2_list = nn.Sequential(*self.UPNet_x2_list)

        self.fuse = nn.Sequential(
            *[
                nn.Conv2d(n_feats * levels, 256, kernel_size=1, padding=0, stride=1),
                nn.ReLU(),
                nn.Conv2d(256, 256, kernel_size=1, padding=0, stride=1),
                nn.ReLU(),
                nn.Conv2d(256, 256, kernel_size=1, padding=0, stride=1),
                nn.ReLU(),
                nn.Conv2d(256, 256, kernel_size=1, padding=0, stride=1),
                nn.ReLU(),
                nn.Conv2d(256, out_channels, kernel_size=1, padding=0, stride=1),
            ]
        )

    def forward(self, x, out_size):
        if type(out_size) == int:
            out_size = [out_size, out_size]

        if type(x) == list:
            return self.forward_list(x, out_size)

        r = torch.tensor([x.shape[2] / out_size[0]]).to(x.device)

        scale_w = self.scale_aware_layer(r.unsqueeze(0))[0]

        x_list = [x]
        for l in range(1, self.levels):
            x_list.append(self.UPNet_x2_list[l - 1](x_list[l - 1]))

        x_resize_list = []
        for l in range(self.levels):
            x_resize = F.interpolate(
                x_list[l], out_size, mode=self.interpolate_mode, align_corners=False
            )
            x_resize *= scale_w[l]
            x_resize_list.append(x_resize)

        out = self.fuse(torch.cat(tuple(x_resize_list), 1))
        return out

    def forward_list(self, h_list, out_size):
        assert (
            len(h_list) == self.levels
        ), "The Length of input list must equal to the number of levels"
        device = h_list[0].device
        r = torch.tensor([h_list[0].shape[2] / out_size[0]], device=device)
        scale_w = self.scale_aware_layer(r.unsqueeze(0))[0]

        x_resize_list = []
        for l in range(self.levels):
            h = h_list[l]
            for i in range(l):
                h = self.UPNet_x2_list[i](h)
            x_resize = F.interpolate(
                h, out_size, mode=self.interpolate_mode, align_corners=False
            )
            x_resize *= scale_w[l]
            x_resize_list.append(x_resize)

        out = self.fuse(torch.cat(tuple(x_resize_list), 1))
        return out


class MLP_Interpolate(nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        self.radius = 1

        self.f_transfer = nn.Sequential(
            *[
                nn.Linear(n_feat * self.radius * self.radius + 2, n_feat),
                nn.ReLU(True),
                nn.Linear(n_feat, 3),
            ]
        )

    def forward(self, x, out_size):
        x_unfold = F.unfold(x, self.radius, padding=self.radius // 2)
        x_unfold = x_unfold.view(
            x.shape[0], x.shape[1] * (self.radius**2), x.shape[2], x.shape[3]
        )

        in_shape = x.shape[-2:]
        in_coord = (
            make_coord(in_shape, flatten=False)
            .cuda()
            .permute(2, 0, 1)
            .unsqueeze(0)
            .expand(x.shape[0], 2, *in_shape)
        )

        if type(out_size) == int:
            out_size = [out_size, out_size]

        out_coord = make_coord(out_size, flatten=True).cuda()
        out_coord = out_coord.expand(x.shape[0], *out_coord.shape)

        q_feat = F.grid_sample(
            x_unfold,
            out_coord.flip(-1).unsqueeze(1),
            mode="nearest",
            align_corners=False,
        )[:, :, 0, :].permute(0, 2, 1)
        q_coord = F.grid_sample(
            in_coord,
            out_coord.flip(-1).unsqueeze(1),
            mode="nearest",
            align_corners=False,
        )[:, :, 0, :].permute(0, 2, 1)

        rel_coord = out_coord - q_coord
        rel_coord[:, :, 0] *= x.shape[-2]
        rel_coord[:, :, 1] *= x.shape[-1]

        inp = torch.cat([q_feat, rel_coord], dim=-1)

        bs, q = out_coord.shape[:2]
        pred = self.f_transfer(inp.view(bs * q, -1)).view(bs, q, -1)
        pred = pred.view(x.shape[0], *out_size, 3).permute(0, 3, 1, 2).contiguous()

        return pred
