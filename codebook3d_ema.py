import torch
import torch.nn as nn
import torch.nn.functional as F

class Codebook3D(nn.Module):
    def __init__(self, args, decay=0.99, eps=1e-5, l2_normalize=False,
                 refresh_every=2000,           
                 refresh_min_count=1.0):       
        super().__init__()
        self.num_codebook_vectors = args.num_codebook_vectors
        self.latent_dim = args.latent_dim
        self.beta = args.beta
        self.decay = decay
        self.eps = eps
        self.l2_normalize = l2_normalize
        self.refresh_every = refresh_every
        self.refresh_min_count = float(refresh_min_count)

        embed = torch.empty(self.num_codebook_vectors, self.latent_dim)
        embed.uniform_(-1.0 / self.num_codebook_vectors, 1.0 / self.num_codebook_vectors)
        if l2_normalize:
            embed = F.normalize(embed, dim=1)
        self.register_buffer("embedding", embed)  # [K,D]

        # EMA 统计量
        self.register_buffer("ema_cluster_size", torch.zeros(self.num_codebook_vectors))       # [K]
        self.register_buffer("ema_embed_avg",   torch.zeros(self.num_codebook_vectors, self.latent_dim))  # [K,D]

        # === 新增: 计步器 ===
        self.register_buffer("step_counter", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _ema_update(self, z_flat, inds):
        K = self.num_codebook_vectors
        device = z_flat.device

        one_hot = torch.zeros(z_flat.size(0), K, device=device)
        one_hot.scatter_(1, inds.unsqueeze(1), 1)

        cluster_size = one_hot.sum(0)           # [K]
        embed_sum = one_hot.t() @ z_flat        # [K,D]

        self.ema_cluster_size.mul_(self.decay).add_(cluster_size, alpha=1 - self.decay)
        self.ema_embed_avg.mul_(self.decay).add_(embed_sum,   alpha=1 - self.decay)

        # 拉普拉斯平滑
        n = self.ema_cluster_size.sum()
        cluster_size = (self.ema_cluster_size + self.eps) / (n + K * self.eps) * n

        embed = self.ema_embed_avg / cluster_size.unsqueeze(1).clamp_min(1e-6)
        if self.l2_normalize:
            embed = F.normalize(embed, dim=1)

        self.embedding.copy_(embed)

    def forward(self, z):
        # [B,C,D,H,W] -> [B,D,H,W,C]
        z = z.permute(0, 2, 3, 4, 1).contiguous()
        z_flattened = z.view(-1, self.latent_dim)  # [N,D]

        emb = self.embedding  # [K,D]

        # 最近邻（L2 或 余弦）
        if self.l2_normalize:
            z_n   = F.normalize(z_flattened, dim=1)
            emb_n = F.normalize(emb, dim=1)
            sims = z_n @ emb_n.t()
            min_encoding_indices = sims.argmax(dim=1)
        else:
            d = (torch.sum(z_flattened ** 2, dim=1, keepdim=True)
                 + torch.sum(emb ** 2, dim=1)
                 - 2 * torch.matmul(z_flattened, emb.t()))
            min_encoding_indices = torch.argmin(d, dim=1)

        z_q = emb.index_select(0, min_encoding_indices).view_as(z)

        # EMA 更新
        self._ema_update(z_flattened.detach(), min_encoding_indices.detach())

        with torch.no_grad():
            self.step_counter += 1
            if self.refresh_every > 0 and int(self.step_counter.item()) % self.refresh_every == 0:
                dead = self.ema_cluster_size < self.refresh_min_count   # [K] 布尔
                if dead.any():
                    k_dead = int(dead.sum().item())
                    idx = torch.randint(0, z_flattened.size(0), (k_dead,), device=z_flattened.device)
                    repl = z_flattened.index_select(0, idx)
                    if self.l2_normalize:
                        repl = F.normalize(repl, dim=1)
                    self.embedding[dead] = repl

                    self.ema_embed_avg[dead] = repl
                    self.ema_cluster_size[dead] = self.refresh_min_count

        # commitment
        commit_loss = torch.mean((z_q.detach() - z) ** 2)
        loss = self.beta * commit_loss

        # STE
        z_q = z + (z_q - z).detach()

        # 还原回形状与索引
        z_q = z_q.permute(0, 4, 1, 2, 3).contiguous()
        B, D, H, W, _ = z.shape
        min_encoding_indices = min_encoding_indices.view(B, D, H, W)
        return z_q, min_encoding_indices, loss
