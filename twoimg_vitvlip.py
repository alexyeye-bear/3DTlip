import torch
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F

# ==== 4D 可学习位置嵌入（注意 dim 要等于 ViT 的 width）====
class FourDPosEmbed(nn.Module):
    """
    分解式 4D 位置：E(d,h,w,c) = Ed[d] + Eh[h] + Ew[w] + Ec[c]
    max_shape 用于限定每个轴的最大长度；forward 时用具体 (D,H,W,C) 做切片。
    返回 [B, D*H*W*C, dim]，展平顺序为 d->h->w->c（最后一维 c 最快）。
    """
    def __init__(self, max_shape, dim):
        super().__init__()
        Dmax, Hmax, Wmax, Cmax = max_shape
        self.ed = nn.Embedding(Dmax, dim)
        self.eh = nn.Embedding(Hmax, dim)
        self.ew = nn.Embedding(Wmax, dim)
        self.ec = nn.Embedding(Cmax, dim)
        nn.init.normal_(self.ed.weight, std=0.02)
        nn.init.normal_(self.eh.weight, std=0.02)
        nn.init.normal_(self.ew.weight, std=0.02)
        nn.init.normal_(self.ec.weight, std=0.02)

    @torch.no_grad()
    def _make_indices(self, D, H, W, C, device):
        return (torch.arange(D, device=device),
                torch.arange(H, device=device),
                torch.arange(W, device=device),
                torch.arange(C, device=device))

    def forward(self, D, H, W, C, B, device):
        d, h, w, c = self._make_indices(D, H, W, C, device)
        pos = ( self.ed(d)[:, None, None, None, :]
              + self.eh(h)[None, :, None, None, :]
              + self.ew(w)[None, None, :, None, :]
              + self.ec(c)[None, None, None, :, :] )         # [D,H,W,C,dim]
        pos = pos.reshape(1, D*H*W*C, -1).expand(B, -1, -1)   # [B,L,dim]
        return pos

# ==== 你已有的 LayerNorm / QuickGELU / ResidualAttentionBlock / Transformer 保持不变 ====
class LayerNorm(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)

class VisionTransformerBLD(nn.Module):
    """
    直接处理序列 token 的 ViT（保留 CLS，仅输出 CLS -> embed_dim）。
    现在支持 4D 位置嵌入：对 token（不含 CLS）加上 4D pos，再进 Transformer。
    """
    def __init__(self,
                 L: int, D_in: int,
                 width: int, layers: int, heads: int, embed_dim: int,
                 shape_4d=(12,4,4,4),     # <- 你的 4D 体素布局
                 max_4d_shape=(12,4,4,4)  # <- 允许的最大范围（可等于 shape_4d）
                 ):
        super().__init__()
        self.L = L
        self.width = width
        self.shape_4d = tuple(shape_4d)
        assert L == int(shape_4d[0] * shape_4d[1] * shape_4d[2] * shape_4d[3]), \
            f"L={L} must equal D*H*W*C={shape_4d}"

        # D_in -> width
        self.proj_in = nn.Identity() if D_in == width else nn.Linear(D_in, width, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))       # CLS token
        # 用一个独立的 cls 位置向量（只给 CLS 用），避免与 token 位置混淆
        self.cls_pos = nn.Parameter(scale * torch.randn(1, 1, width))

        self.ln_pre = LayerNorm(width)
        self.transformer = Transformer(width=width, layers=layers, heads=heads)
        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, embed_dim))       # CLS -> embed_dim

        # ✅ 4D 位置嵌入维度要等于 width（因为是在 Transformer 之前相加）
        self.pos4d = FourDPosEmbed(max_4d_shape, dim=width)

    def forward(self, tokens: torch.Tensor):
        """
        tokens: [B, L, D_in]
        return: [B, embed_dim]  （CLS 向量，用于对比）
        """
        B, L, _ = tokens.shape
        assert L == self.L, f"Expected L={self.L}, got {L}"
        D, H, W, C = self.shape_4d

        x_tok = self.proj_in(tokens)                         # [B, L, width]

        # 4D 位置编码（只加到 token，不加到 CLS）
        pos_tok = self.pos4d(D, H, W, C, B=B, device=x_tok.device)  # [B, L, width]
        x_tok = x_tok + pos_tok

        # 拼 CLS
        cls = self.class_embedding[None, None, :].expand(B, 1, self.width)    # [B,1,width]
        x = torch.cat([cls, x_tok], dim=1)                                    # [B, L+1, width]
        x = x + self.cls_pos.to(x.dtype)                                      # 只给 CLS 一个位置向量

        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)                                                # NLD -> LND
        x = self.transformer(x)                                                # [L+1,B,width]
        x = x.permute(1, 0, 2)                                                # [B,L+1,width]

        x = self.ln_post(x[:, 0, :])                                          # 取 CLS
        x = x @ self.proj                                                     # [B, embed_dim]
        return x

class DualViTCLIP_BLD(nn.Module):
    """
    两个头都是 ViT encoder；支持 TokLIP 风格：
    indices_embed [B,D,H,W,C,code_dim] --(flatten+MLP)--> tokens [B,L,width] --ViT--> [B,embed_dim]
    """
    def __init__(self,
                 L: int,
                 # 这里把 ViT 的 D_in 固定为 width（因为我们前置 MLP已经投到 width）
                 width_a: int, width_b: int,
                 layers_a: int, layers_b: int,
                 heads_a: int, heads_b: int,
                 embed_dim: int,
                 shape_4d=(12,4,4,4),
                 code_dim_a: int = 256,   # ← indices_embed 的最后一维
                 code_dim_b: int = 256,
                 init_logit_scale: float = 1/0.07):
        super().__init__()

        # === TokLIP式前置 MLP：code_dim -> 4*width -> width ===
        self.pre_mlp_a = nn.Sequential(
            nn.Linear(code_dim_a, 4*width_a),
            nn.GELU(),
            nn.Linear(4*width_a, width_a),
        )
        self.pre_mlp_b = nn.Sequential(
            nn.Linear(code_dim_b, 4*width_b),
            nn.GELU(),
            nn.Linear(4*width_b, width_b),
        )

        # === ViT 编码器：D_in=width（前置 MLP 后的维度），proj_in=Identity ===
        # self.backbone = VisionTransformerBLD(
        #     L=L, D_in=width_a, width=width_a,
        #     layers=layers_a, heads=heads_a, embed_dim=embed_dim,
        #     shape_4d=shape_4d, max_4d_shape=shape_4d
        # )
        
        self.enc_a = VisionTransformerBLD(
            L=L, D_in=width_a, width=width_a,
            layers=layers_a, heads=heads_a, embed_dim=embed_dim,
            shape_4d=shape_4d, max_4d_shape=shape_4d
        )
        self.enc_b = VisionTransformerBLD(
            L=L, D_in=width_b, width=width_b,
            layers=layers_b, heads=heads_b, embed_dim=embed_dim,
            shape_4d=shape_4d, max_4d_shape=shape_4d
        )

        self.logit_scale = nn.Parameter(torch.log(torch.tensor(float(init_logit_scale))))
        
        # 两个独立小头（不共享）
        self.proj_a = nn.Sequential(nn.Linear(embed_dim, embed_dim, bias=False))
        self.proj_b = nn.Sequential(nn.Linear(embed_dim, embed_dim, bias=False))

    # ---- indices_embed -> tokens (TokLIP MLP) ----
    # 输入 x: [B, D, H, W, C, code_dim]，其中 (D,H,W,C) 必须与 shape_4d 一致 (12,4,4,4)；输出 [B, L, width]
    def preprocess_indices_embed_a(self, x: torch.Tensor) -> torch.Tensor:
        B, D, H, W, C, Cd = x.shape
        tokens = x.view(B, D*H*W*C, Cd)           # [B, L, code_dim]
        tokens = self.pre_mlp_a(tokens)           # [B, L, width_a]
        return tokens

    def preprocess_indices_embed_b(self, x: torch.Tensor) -> torch.Tensor:
        B, D, H, W, C, Cd = x.shape
        tokens = x.view(B, D*H*W*C, Cd)           # [B, L, code_dim]
        tokens = self.pre_mlp_b(tokens)           # [B, L, width_b]
        return tokens

    def _l2(self, x, dim=1, eps=1e-6):
        # 保留梯度的 L2 归一化
        return F.normalize(x, p=2, dim=dim, eps=eps)

    # 仍保留直接喂 tokens 的接口（你已有的数据流不受影响）
    def encode_a(self, tokens_a: torch.Tensor):
        # return self.enc_a(tokens_a)
        h = self.enc_a(tokens_a)
        z = self.proj_a(h)
        return z

    def encode_b(self, tokens_b: torch.Tensor):
        # return self.enc_b(tokens_b)
        h = self.enc_b(tokens_b)
        z = self.proj_b(h)
        return z

    # 新增：直接喂 indices_embed 的便捷接口
    def encode_a_from_indices(self, indices_embed_a: torch.Tensor):
        tokens_a = self.preprocess_indices_embed_a(indices_embed_a)
        return self.enc_a(tokens_a)

    def encode_b_from_indices(self, indices_embed_b: torch.Tensor):
        tokens_b = self.preprocess_indices_embed_b(indices_embed_b)
        return self.enc_b(tokens_b)

    # forward 仍然接收 tokens；若你想直接接 indices_embed，可自己在外面先 preprocess 再传进来
    def forward(self, tokens_a, tokens_b):
        za = self.encode_a(tokens_a)             # [B, embed_dim]
        zb = self.encode_b(tokens_b)             # [B, embed_dim]
        za = self._l2(za); zb = self._l2(zb)

        s = self.logit_scale.exp()
        logits_a2b = s * (za @ zb.t())        # [B, B]
        logits_b2a = logits_a2b.t()
        return logits_a2b, logits_b2a, (za, zb)


if __name__ == "__main__":
    import torch
    import torch.nn.functional as F

    torch.manual_seed(0)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    shape_4d = (12, 4, 4, 4)
    L = 12 * 4 * 4 * 4   # 768
    B = 4
    code_dim = 256       # indices_embed 的最后一维

    width_a = width_b = 256
    layers_a = layers_b = 8
    heads_a = heads_b = 4
    embed_dim = 512
    lr = 1e-4

    model = DualViTCLIP_BLD(
        L=L,
        width_a=width_a, width_b=width_b,
        layers_a=layers_a, layers_b=layers_b,
        heads_a=heads_a, heads_b=heads_b,
        embed_dim=embed_dim,
        shape_4d=shape_4d,
        code_dim_a=code_dim, code_dim_b=code_dim
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    # 模拟 TokLIP 的“indices_embed”： [B, D, H, W, C, 256]
    indices_embed_a = torch.randn(B, *shape_4d, code_dim, device=device)
    indices_embed_b = torch.randn(B, *shape_4d, code_dim, device=device)

    # 通过 TokLIP 式 MLP 转成 ViT 输入 tokens
    tokens_a = model.preprocess_indices_embed_a(indices_embed_a)   # [B, L, width_a]
    tokens_b = model.preprocess_indices_embed_b(indices_embed_b)   # [B, L, width_b]

    # 前向
    logits_ab, logits_ba, (za, zb) = model(tokens_a, tokens_b)
    print("tokens_a:", tokens_a.shape, "tokens_b:", tokens_b.shape)      # [B,768,768]
    print("logits:", logits_ab.shape, "za:", za.shape, "zb:", zb.shape)  # [B,B], [B,512]

    # InfoNCE
    tgt = torch.arange(B, device=device)
    loss = 0.5 * (F.cross_entropy(logits_ab, tgt) + F.cross_entropy(logits_ba, tgt))
    print("loss:", float(loss))

    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
