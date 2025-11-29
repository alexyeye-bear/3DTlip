import torch.nn as nn
import torch
from helper_3d import ResidualBlock3D, NonLocalBlock3D, UpSampleBlock3D, GroupNorm3D, Swish

class Decoder3D(nn.Module):
    def __init__(self, args):
        super(Decoder3D, self).__init__()
        attn_resolutions = [16]
        ch_mult = [64, 128, 256, 256, 512]
        num_resolutions = len(ch_mult)
        block_in = ch_mult[num_resolutions - 1]
        curr_res = 64 // 2 ** (num_resolutions - 1)  # 输入为 64×64×64 的 latent

        layers = [nn.Conv3d(args.latent_dim, block_in, kernel_size=3, stride=1, padding=1),
                  ResidualBlock3D(block_in, block_in),
                  NonLocalBlock3D(block_in),
                  ResidualBlock3D(block_in, block_in)]

        for i in reversed(range(num_resolutions)):
            block_out = ch_mult[i]
            for i_block in range(3):
                layers.append(ResidualBlock3D(block_in, block_out))
                block_in = block_out
                # if curr_res in attn_resolutions:
                #     layers.append(NonLocalBlock3D(block_in))
            if i > 1:
                layers.append(UpSampleBlock3D(block_in))
                curr_res *= 2

        layers.append(GroupNorm3D(block_in))
        # layers.append(Swish())
        layers.append(nn.Conv3d(block_in, args.image_channels, kernel_size=3, stride=1, padding=1))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class Args:
    latent_dim = 256
    image_channels = 1 

if __name__ == "__main__":
    args = Args()
    args.device = torch.device("cuda:7")

    model = Decoder3D(args).to(args.device)
    dummy_input = torch.randn(2, args.latent_dim, 7, 8, 7).to(args.device)

    with torch.no_grad():
        output = model(dummy_input)

    print("输出 shape:", output.shape)
