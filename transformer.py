import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from bidirectional_transformer import BidirectionalTransformer
from types import SimpleNamespace
_CONFIDENCE_OF_KNOWN_TOKENS = torch.Tensor([torch.inf]).to("cuda")


class VQGANTransformer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.num_image_tokens = args.num_image_tokens
        self.sos_token = args.num_codebook_vectors + 1
        self.mask_token_id = args.num_codebook_vectors
        self.choice_temperature = 4.5

        self.gamma = self.gamma_func("cosine")

        # self.transformer = BidirectionalTransformer(
        #                         patch_size=8, embed_dim=args.dim, depth=args.n_layers, num_heads=12, mlp_ratio=4, qkv_bias=True,
        #                         norm_layer=partial(nn.LayerNorm, eps=1e-6), vocab_size=8192+1)
        self.transformer = BidirectionalTransformer(args)
        self.vqgan = self.load_vqgan(args)
        # print(f"Transformer parameters: {sum([p.numel() for p in self.transformer.parameters()])}")

    # def load_checkpoint(self, epoch):
    #     self.load_state_dict(torch.load(os.path.join("checkpoints", f"transformer_epoch_{epoch}.pt")))
    #     print("Check!")

    @staticmethod
    def load_vqgan(transf_args):
        from vit_vqgan import VQGAN
        # 用 SimpleNamespace 快速创建一个带属性的 args 对象
        # 这里得用vqgan 的参数
        args = SimpleNamespace(
            latent_dim=transf_args.vq_latentdim,  
            image_size=transf_args.vq_image_size,
            num_codebook_vectors=transf_args.num_codebook_vectors,
            image_channels=transf_args.image_channels,
            device=transf_args.device,
            patch_size=transf_args.vq_patch_size,
            transformer_layers=transf_args.vq_transformer_layers,
            transformer_heads=transf_args.vq_transformer_heads,
            transformer_dim_head=transf_args.vq_transformer_dim_head,
            transformer_mlp_mult=transf_args.vq_transformer_mlp_mult,
            beta=0.25
        )
        model = VQGAN(args)
        state_dict = torch.load(transf_args.vitvq_ckpt_path, map_location=transf_args.device)
        model.load_state_dict(state_dict)
        
        # 冻结模型所有参数，禁止计算梯度
        for param in model.parameters():
            param.requires_grad = False
        
        model = model.eval()
        return model

    @torch.no_grad()
    def encode_to_z(self, x):
        # quant_z, indices, _ = self.vqgan.encode(x)
        quant_z, _, (_, _, indices) = self.vqgan.encode(x)
        indices = indices.view(quant_z.shape[0], -1)
        return quant_z, indices

    def forward(self, x):
        # _, z_indices = self.encode_to_z(x)
        #
        # r = np.random.uniform()
        # mask = torch.bernoulli(r * torch.ones(z_indices.shape[-1], device=z_indices.device))
        # mask = mask.round().bool()
        #
        # target = z_indices[:, mask]
        #
        # logits = self.transformer(z_indices, mask)

        # 训练时我这里直接加载 quant,不用从头变了
        # _, z_indices = self.encode_to_z(x)
        
        B=x.shape[0]
        z_indices = x.view(B, -1)
        sos_tokens = torch.ones(x.shape[0], 1, dtype=torch.long, device=z_indices.device) * self.sos_token

        r = math.floor(self.gamma(np.random.uniform()) * z_indices.shape[1])
        sample = torch.rand(z_indices.shape, device=z_indices.device).topk(r, dim=1).indices
        mask = torch.zeros(z_indices.shape, dtype=torch.bool, device=z_indices.device)
        mask.scatter_(dim=1, index=sample, value=True)

        # torch.rand(z_indices.shape, device=z_indices.device)
        # mask = torch.bernoulli(r * torch.ones(z_indices.shape, device=z_indices.device))
        # mask = torch.bernoulli(torch.rand(z_indices.shape, device=z_indices.device))
        # mask = mask.round().to(dtype=torch.int64)
        # masked_indices = torch.zeros_like(z_indices)
        masked_indices = self.mask_token_id * torch.ones_like(z_indices, device=z_indices.device)
        a_indices = mask * z_indices + (~mask) * masked_indices

        a_indices = torch.cat((sos_tokens, a_indices), dim=1)

        target = torch.cat((sos_tokens, z_indices), dim=1)

        logits = self.transformer(a_indices)

        return logits, target

    def top_k_logits(self, logits, k):
        v, ix = torch.topk(logits, k)
        out = logits.clone()
        if k == 0:
            out[:, :] = self.sos_token
        else:
            out[out < v[..., [-1]]] = self.sos_token
        return out

    def gamma_func(self, mode="cosine"):
        if mode == "linear":
            return lambda r: 1 - r
        elif mode == "cosine":
            return lambda r: np.cos(r * np.pi / 2)
        elif mode == "square":
            return lambda r: 1 - r ** 2
        elif mode == "cubic":
            return lambda r: 1 - r ** 3
        else:
            raise NotImplementedError

    def create_input_tokens_normal(self, num, label=None):
        # label_tokens = label * torch.ones([num, 1])
        # Shift the label by codebook_size
        # label_tokens = label_tokens + self.vqgan.codebook.num_codebook_vectors
        # Create blank masked tokens
        blank_tokens = torch.ones((num, self.num_image_tokens), device="cuda")
        masked_tokens = self.mask_token_id * blank_tokens
        # Concatenate the two as input_tokens
        # input_tokens = torch.concat([label_tokens, masked_tokens], dim=-1)
        # return input_tokens.to(torch.int32)
        return masked_tokens.to(torch.int64)

    def tokens_to_logits(self, seq):
        logits = self.transformer(seq)
        # logits = logits[..., :self.vqgan.codebook.num_codebook_vectors]  # why is maskgit returning [8, 257, 2025]?
        return logits

    def mask_by_random_topk(self, mask_len, probs, temperature=1.0):
        confidence = torch.log(probs) + temperature * torch.distributions.gumbel.Gumbel(0, 1).sample(probs.shape).to("cuda")
        sorted_confidence, _ = torch.sort(confidence, dim=-1)
        # Obtains cut off threshold given the mask lengths.
        cut_off = torch.take_along_dim(sorted_confidence, mask_len.to(torch.long), dim=-1)
        # Masks tokens with lower confidence.
        masking = (confidence < cut_off)
        return masking

    @torch.no_grad()
    def sample_good(self, inputs=None, num=1, T=11, mode="cosine"):
        # self.transformer.eval()
        N = self.num_image_tokens
        if inputs is None:
            inputs = self.create_input_tokens_normal(num)
        else:
            inputs = torch.hstack(
                (inputs, torch.zeros((inputs.shape[0], N - inputs.shape[1]), device="cuda", dtype=torch.int).fill_(self.mask_token_id)))

        sos_tokens = torch.ones(inputs.shape[0], 1, dtype=torch.long, device=inputs.device) * self.sos_token
        inputs = torch.cat((sos_tokens, inputs), dim=1)

        unknown_number_in_the_beginning = torch.sum(inputs == self.mask_token_id, dim=-1)
        gamma = self.gamma_func(mode)
        cur_ids = inputs  # [8, 257]
        for t in range(T):
            logits = self.tokens_to_logits(cur_ids)  # call transformer to get predictions [8, 257, 1024]
            sampled_ids = torch.distributions.categorical.Categorical(logits=logits).sample()

            unknown_map = (cur_ids == self.mask_token_id)  # which tokens need to be sampled -> bool [8, 257]
            sampled_ids = torch.where(unknown_map, sampled_ids, cur_ids)  # replace all -1 with their samples and leave the others untouched [8, 257]

            ratio = 1. * (t + 1) / T  # just a percentage e.g. 1 / 12
            mask_ratio = gamma(ratio)

            probs = F.softmax(logits, dim=-1)  # convert logits into probs [8, 257, 1024]
            selected_probs = torch.squeeze(torch.take_along_dim(probs, torch.unsqueeze(sampled_ids, -1), -1), -1)  # get probability for selected tokens in categorical call, also for already sampled ones [8, 257]

            selected_probs = torch.where(unknown_map, selected_probs, _CONFIDENCE_OF_KNOWN_TOKENS)  # ignore tokens which are already sampled [8, 257]

            mask_len = torch.unsqueeze(torch.floor(unknown_number_in_the_beginning * mask_ratio), 1)  # floor(256 * 0.99) = 254 --> [254, 254, 254, 254, ....]
            mask_len = torch.maximum(torch.zeros_like(mask_len), torch.minimum(torch.sum(unknown_map, dim=-1, keepdim=True)-1, mask_len))  # add -1 later when conditioning and also ones_like. Zeroes just because we have no cond token
            # max(1, min(how many unknown tokens, how many tokens we want to sample))

            # Adds noise for randomness
            masking = self.mask_by_random_topk(mask_len, selected_probs, temperature=self.choice_temperature * (1. - ratio))
            # Masks tokens with lower confidence.
            cur_ids = torch.where(masking, self.mask_token_id, sampled_ids)
            # print((cur_ids == 8192).count_nonzero())

        # self.transformer.train()
        return cur_ids[:, 1:]

    @torch.no_grad()
    def log_images(self, x, mode="cosine"):
        log = dict()

        _, z_indices = self.encode_to_z(x)

        # create new sample
        index_sample = self.sample_good(mode=mode)
        x_new = self.indices_to_image(index_sample)

        # create a "half" sample
        z_start_indices = z_indices[:, :z_indices.shape[1] // 2]
        half_index_sample = self.sample_good(z_start_indices, mode=mode)
        x_sample = self.indices_to_image(half_index_sample)

        # create reconstruction
        x_rec = self.indices_to_image(z_indices)

        log["input"] = x
        log["rec"] = x_rec
        log["half_sample"] = x_sample
        log["new_sample"] = x_new
        return log, torch.concat((x, x_rec, x_sample, x_new))

    def indices_to_image(self, indices, p1=32, p2=32,latent_dim=32):
        ix_to_vectors = self.vqgan.codebook.embedding(indices).reshape(indices.shape[0], p1, p2, latent_dim)
        # ix_to_vectors = self.vqgan.quantize.embedding(indices).reshape(indices.shape[0], 16, 16, 256)
        ix_to_vectors = ix_to_vectors.permute(0, 3, 1, 2)
        image = self.vqgan.decode(ix_to_vectors)
        return image

    @staticmethod
    def create_masked_image(image: torch.Tensor, x_start: int = 100, y_start: int = 100, size: int = 50):
        mask = torch.ones_like(image, dtype=torch.int)
        mask[:, :, x_start:x_start + size, y_start:y_start + size] = 0
        return image * mask, mask
    
    # 直接从vq开始的补全
    def sample_inpainting(self, vq_indices, mask_rate=0.5, seed=42, max_iters=10, fill_fraction=0.2):
        """
        Iteratively inpaint VQ indices using a Transformer model compatible with the provided training forward.
        
        Args:
            model:  self.transformer(transformer_model), expects input seq shape [B, 1+N] and returns logits [B, 1+N, C].
                Must have attributes `sos_token` and `mask_token_id`.
            vq_indices (Tensor): shape [B, H, W], int indices in [0, C)
            mask_rate (float): fraction of patches to mask (e.g., 0.5 for 50%)
            seed (int): random seed for reproducibility
            max_iters (int): number of iterative sampling rounds
            fill_fraction (float): fraction of remaining masked positions to fill each iteration
        Returns:
            Tensor of shape [B, H, W] with inpainted indices
        """
        # torch.manual_seed(seed)
        B, H, W = vq_indices.shape
        N = H * W
        device = vq_indices.device

        # Flatten indices
        indices_flat = vq_indices.view(B, N).clone()

        # Create random mask per batch
        mask = torch.zeros(B, N, dtype=torch.bool, device=device)
        num_mask = int(mask_rate * N)
        # print('num_mask',num_mask)
        for i in range(B):
            perm = torch.randperm(N, device=device)
            mask[i, perm[:num_mask]] = True
            
        before_mask = mask.clone()
        
        # num_trues = before_mask[0].sum().item()
        # print(f"transformer 里 True 的个数：{num_trues}")

        # Initialize masked input for the model
        a_indices_base = indices_flat.clone()
        a_indices_base[mask] = self.mask_token_id

        # SOS token prepended to every sequence
        sos_tokens = torch.full((B, 1), self.sos_token, dtype=torch.long, device=device)

        # Iterative fill
        for _ in range(max_iters):
            # Build input sequence [SOS | masked_and_filled_tokens]
            input_seq = torch.cat((sos_tokens, a_indices_base), dim=1)  # shape [B,1+N]
            logits = self.transformer(input_seq)                                   # [B,1+N,C]
            # Focus logits on token positions (exclude SOS)
            token_logits = logits[:, 1:, :]                             # [B, N, C]
            probs = F.softmax(token_logits, dim=-1)                     # [B, N, C]
            confidences, preds = probs.max(dim=-1)                      # [B, N], scores and predicted tokens

            if not mask.any():
                break  # all filled

            # Fill top-k confident positions per sample
            for i in range(B):
                unfilled = mask[i]
                if not unfilled.any():
                    continue
                conf_unfilled = confidences[i][unfilled]
                k = max(1, int(conf_unfilled.numel() * fill_fraction))
                topk_vals, topk_idx = torch.topk(conf_unfilled, k)
                candidate_positions = torch.nonzero(unfilled, as_tuple=False).view(-1)
                fill_positions = candidate_positions[topk_idx]
                a_indices_base[i, fill_positions] = preds[i, fill_positions]
                mask[i, fill_positions] = False

            # Final fill for any remaining
            if mask.any():
                input_seq = torch.cat((sos_tokens, a_indices_base), dim=1)
                logits = self.transformer(input_seq)
                final_preds = F.softmax(logits[:, 1:, :], dim=-1).argmax(dim=-1)
                indices_flat[mask] = final_preds[mask]

        return indices_flat.view(B, H, W),before_mask.view(B, H, W)
    
    def sample_inpaint_timemask(self, vq_indices, mask_rate=0.5, max_iters=10, fill_fraction=0.2):
        """
        Iteratively inpaint VQ indices by masking a randomly chosen contiguous block of 5 columns (4 to 20) in each run.

        Args:
            vq_indices (Tensor): shape [B, H, W]
        Returns:
            Tensor of shape [B, H, W] with inpainted indices,
            and Tensor [B, H, W] boolean mask indicating masked positions
        """

        B, H, W = vq_indices.shape
        device = vq_indices.device
        N = H * W

        # Flatten indices
        indices_flat = vq_indices.view(B, N).clone()

        # Randomly select starting column for a block of 5 contiguous columns between 4 and 20 inclusive
        # Last valid start column is 20-5+1 = 16
        start_col = torch.randint(4, 20, (1,), device=device).item()
        cols = list(range(start_col, start_col + 1))

        # Create mask for those columns
        mask = torch.zeros(B, N, dtype=torch.bool, device=device)
        for c in cols:
            idxs = torch.arange(c, N, W, device=device)
            mask[:, idxs] = True
        before_mask = mask.clone()

        # Initialize masked input for model
        a_indices = indices_flat.clone()
        a_indices[mask] = self.mask_token_id
        sos_tokens = torch.full((B, 1), self.sos_token, dtype=torch.long, device=device)

        # 记录初始每个样本的被 mask 总数 M0
        M0 = before_mask.sum(dim=1)   # Tensor([M0_0, M0_1, …], device=device)

        # —— 迭代 inpainting —— #
        for t in range(max_iters):
            # 调度比率 r = t/T
            r = t / float(max_iters)
            gamma = 0.5 * (1 + math.cos(math.pi * r))

            # 模型前向，获取每个位置的置信度和预测
            input_seq = torch.cat((sos_tokens, a_indices), dim=1)
            logits = self.transformer(input_seq)           # [B, N+1, V]
            token_logits = logits[:, 1:, :]               # [B, N, V]
            probs = F.softmax(token_logits, dim=-1)       # [B, N, V]
            confidences, preds = probs.max(dim=-1)        # [B, N]

            # 对每个样本分别按最新 mask 和调度填充 top-​k
            for i in range(B):
                M_curr = mask[i].sum().item()             # 当前还要填的个数
                n_keep = int((M0[i].float() * gamma).ceil().item())  # 本轮还要保留的数量
                k = M_curr - n_keep                       # 本轮要填多少个
                if k <= 0:
                    continue

                # 在当前 mask 位置中，选置信度最高的 k 个填入预测
                unfilled_idx = torch.nonzero(mask[i], as_tuple=False).view(-1)  # 所有待填位置
                conf_unfilled = confidences[i][mask[i]]                        # [M_curr]
                topk = torch.topk(conf_unfilled, k).indices                    # [k]
                fill_pos = unfilled_idx[topk]                                  # [k]

                a_indices[i, fill_pos] = preds[i, fill_pos]
                mask[i, fill_pos] = False

            remaining = mask.sum().item()
            # print(f"Iter {t+1}/{max_iters}, remaining masked: {remaining}")
            if remaining == 0:
                break

        # 最后一轮还没填完的，全量 argmax
        if mask.any():
            input_seq = torch.cat((sos_tokens, a_indices), dim=1)
            final_logits = self.transformer(input_seq)[:, 1:, :]
            final_preds = F.softmax(final_logits, dim=-1).argmax(dim=-1)
            indices_flat[mask] = final_preds[mask]

        return a_indices.view(B, H, W), before_mask.view(B, H, W)


