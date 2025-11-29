import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock3D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.block = nn.Sequential(
            GroupNorm3D(in_channels),
            Swish(),
            nn.Conv3d(in_channels, out_channels, 3, 1, 1),
            GroupNorm3D(out_channels),
            Swish(),
            nn.Conv3d(out_channels, out_channels, 3, 1, 1)
        )
        if in_channels != out_channels:
            self.channel_up = nn.Conv3d(in_channels, out_channels, 1)

    def forward(self, x):
        if self.in_channels != self.out_channels:
            return self.block(x) + self.channel_up(x)
        else:
            return x + self.block(x)

class DownSampleBlock3D(nn.Module):
    def __init__(self, channels):
        super(DownSampleBlock3D, self).__init__()
        self.conv = nn.Conv3d(channels, channels, 3, 2, 1)  # padding 1 更合理

    def forward(self, x):
        return self.conv(x)

class UpSampleBlock3D(nn.Module):
    def __init__(self, channels):
        super(UpSampleBlock3D, self).__init__()
        self.conv = nn.Conv3d(channels, channels, 3, 1, 1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2., mode='trilinear', align_corners=False)
        return self.conv(x)

class NonLocalBlock3D(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.norm = GroupNorm3D(in_channels)
        self.q = nn.Conv3d(in_channels, in_channels, 1)
        self.k = nn.Conv3d(in_channels, in_channels, 1)
        self.v = nn.Conv3d(in_channels, in_channels, 1)
        self.proj_out = nn.Conv3d(in_channels, in_channels, 1)

    def forward(self, x):
        h_ = self.norm(x)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        b, c, d, h, w = q.shape
        q = q.reshape(b, c, -1).permute(0, 2, 1)  # [B, D*H*W, C]
        k = k.reshape(b, c, -1)                  # [B, C, D*H*W]
        v = v.reshape(b, c, -1)                  # [B, C, D*H*W]

        attn = torch.bmm(q, k)                   # [B, D*H*W, D*H*W]
        attn = attn * (c ** -0.5)
        attn = F.softmax(attn, dim=-1)

        attn = attn.permute(0, 2, 1)             # [B, D*H*W, D*H*W]
        out = torch.bmm(v, attn)                 # [B, C, D*H*W]
        out = out.view(b, c, d, h, w)

        out = self.proj_out(out)
        return x + out

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class GroupNorm3D(nn.Module):
    def __init__(self, in_channels):
        super(GroupNorm3D, self).__init__()
        self.gn = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)

    def forward(self, x):
        return self.gn(x)