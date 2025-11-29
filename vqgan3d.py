import torch
import torch.nn as nn
from encoder3d import Encoder3D
from decoder3d import Decoder3D
from codebook3d_ema import Codebook3D  

class VQGAN3D(nn.Module):
    def __init__(self, args):
        super(VQGAN3D, self).__init__()
        self.encoder = Encoder3D(args).to(args.device)
        self.decoder = Decoder3D(args).to(args.device)
        self.codebook = Codebook3D(args).to(args.device)
        self.quant_conv = nn.Conv3d(args.latent_dim, args.latent_dim, 1).to(args.device)
        self.post_quant_conv = nn.Conv3d(args.latent_dim, args.latent_dim, 1).to(args.device)

    def forward(self, imgs):
        encoded_images = self.encoder(imgs)
        quantized_encoded_images = self.quant_conv(encoded_images)

        # print('vq大小:', quantized_encoded_images.shape)
        codebook_mapping, codebook_indices, q_loss = self.codebook(quantized_encoded_images)
        quantized_codebook_mapping = self.post_quant_conv(codebook_mapping)
        # print('quantized_codebook_mapping 大小:', quantized_codebook_mapping.shape)
        decoded_images = self.decoder(quantized_codebook_mapping)

        return decoded_images, codebook_indices, q_loss

    def encode(self, x):
        encoded_images = self.encoder(x)
        quantized_encoded_images = self.quant_conv(encoded_images)
        codebook_mapping, codebook_indices, q_loss = self.codebook(quantized_encoded_images)
        return codebook_mapping, codebook_indices, q_loss

    def decode(self, z):
        quantized_codebook_mapping = self.post_quant_conv(z)
        decoded_images = self.decoder(quantized_codebook_mapping)
        return decoded_images

    def calculate_lambda(self, nll_loss, g_loss):
        last_layer = self.decoder.model[-1]
        last_layer_weight = last_layer.weight
        nll_grads = torch.autograd.grad(nll_loss, last_layer_weight, retain_graph=True)[0]
        g_grads = torch.autograd.grad(g_loss, last_layer_weight, retain_graph=True)[0]

        λ = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        λ = torch.clamp(λ, 0, 1e4).detach()
        return 0.8 * λ

    @staticmethod
    def adopt_weight(disc_factor, i, threshold, value=0.):
        if i < threshold:
            disc_factor = value
        return disc_factor

    def load_checkpoint(self, path):
        self.load_state_dict(torch.load(path))
        print("Loaded 3D Checkpoint for VQGAN....")

class Args:
    image_channels = 1         # 如 1 表示单通道 3D 数据
    latent_dim = 256
    num_codebook_vectors = 1024
    device = torch.device('cuda:7') if torch.cuda.is_available() else torch.device('cpu')
    beta = 0.25

if __name__ == "__main__":
    args = Args()
    model = VQGAN3D(args).to(args.device)

    # 构造 3D 输入数据：[B, C, D, H, W]
    dummy_input = torch.randn(2, args.image_channels, 64, 64, 64).to(args.device)

    # 推理
    with torch.no_grad():
        out, indices, qloss = model(dummy_input)

    print("\n📥 输入大小:", dummy_input.shape)
    print("📤 输出大小:", out.shape)
    print("📌 codebook indices 形状:", indices.shape)
    print("📉 quantization loss:", qloss.item())
