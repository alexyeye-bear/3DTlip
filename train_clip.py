import os
import argparse
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import utils as vutils
from discriminator3d import Discriminator3D
# from lpips import LPIPS
import lpips
from dataloader_abcd3d_oldway import weights_init PairWindowDataset

from vqgan3d_origin import VQGAN3D
from torch.utils.tensorboard import SummaryWriter
import torchvision
from datetime import datetime
import csv
from torch.utils.data import Dataset, DataLoader
from twoimg_vitvlip import DualViTCLIP_BLD

def contrastive_loss(logits_ab: torch.Tensor, logits_ba: torch.Tensor):
    """
    logits_*: [B, B]
    return: loss (scalar), top1_acc (float)
    """
    B = logits_ab.size(0)
    if B < 2:
        # batch太小，对比损失没意义
        return logits_ab.new_tensor(0.0, requires_grad=True), 0.0

    targets = torch.arange(B, device=logits_ab.device)
    loss_ab = F.cross_entropy(logits_ab, targets)   # a->b
    loss_ba = F.cross_entropy(logits_ba, targets)   # b->a
    loss = 0.5 * (loss_ab + loss_ba)

    # 计算 top-1 acc（双向平均）
    pred_ab = logits_ab.argmax(dim=1)
    pred_ba = logits_ba.argmax(dim=1)
    acc = 0.5 * ((pred_ab == targets).float().mean() + (pred_ba == targets).float().mean())
    return loss, acc.item()

@torch.no_grad()
def evaluate_clip(model, vqgan, loader, device="cuda", use_amp=False):
    model.eval()
    total_loss, total_acc, n = 0.0, 0.0, 0
    for xa, xb, _ in loader:
        xa = xa.to(device, non_blocking=True)   # [B,k,D,H,W] long
        xb = xb.to(device, non_blocking=True)

        # 只查表（VQGAN 冻结）
        indices_embed_a = vqgan.codebook.embedding(xa)
        indices_embed_b = vqgan.codebook.embedding(xb)

        # TokLIP 式 MLP -> tokens
        tokens_a = model.preprocess_indices_embed_a(indices_embed_a)
        tokens_b = model.preprocess_indices_embed_b(indices_embed_b)

        # 前向 + 对比损失
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits_ab, logits_ba, _ = model(tokens_a, tokens_b)
            loss, acc = contrastive_loss(logits_ab, logits_ba)

        total_loss += loss.item()
        total_acc  += acc
        n += 1
        
        if n==4:
            break

    avg_loss = total_loss / max(n, 1)
    avg_acc  = total_acc  / max(n, 1)
    return avg_loss, avg_acc

class TrainVQGAN:
    def __init__(self, args):
        self.vqgan = VQGAN3D(args)
        
        state_dict = torch.load(args.ckpt_path, map_location=args.device)
        self.vqgan.load_state_dict(state_dict)
        
        self.vqgan.eval()
        
        self.discriminator = Discriminator3D(args).to(device=args.device)
        self.discriminator.apply(weights_init)
        
        # self.perceptual_loss = LPIPS().eval().to(device=args.device)
        self.perceptual_loss = lpips.LPIPS(net='vgg').to(device=args.device)
        
        self.opt_vq, self.opt_disc = self.configure_optimizers(args)

        # 获取当前时间并格式化
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        dir_path = os.path.join(args.base_dir,args.expr_name, timestamp)
        args.result_dir = os.path.join(dir_path, 'result')
        args.log_dir = os.path.join(dir_path, 'log')
        args.checkd_dir = os.path.join(dir_path, 'checks')
        
        bmask_np=np.load('/mni_x32_np.npy')
        
        assert bmask_np.shape == (32, 32, 32), f"mask shape {bmask_np.shape} != (32,32,32)"

        bmask_bool = (bmask_np > 0.2)                      # numpy.bool_ 数组
        bmask_t = torch.from_numpy(bmask_bool)             # dtype=torch.bool，shape [32,32,32]
        bmask_t = bmask_t.unsqueeze(0).unsqueeze(0)        # -> [1,1,32,32,32]

        self.bmask_t=bmask_t.to(args.device)

        self.prepare_training(args)

        self.train(args)

    @staticmethod
    def prepare_training(self):
        os.makedirs(args.result_dir, exist_ok=True)
        os.makedirs(args.checkd_dir, exist_ok=True)

    def configure_optimizers(self, args):
        lr = args.learning_rate
        opt_vq = torch.optim.Adam(list(self.vqgan.encoder.parameters()) +
                                  list(self.vqgan.decoder.parameters()) +
                                  list(self.vqgan.codebook.parameters()) +
                                  list(self.vqgan.quant_conv.parameters()) +
                                  list(self.vqgan.post_quant_conv.parameters()),
                                  lr=lr, eps=1e-08, betas=(args.beta1, args.beta2))
        opt_disc = torch.optim.Adam(self.discriminator.parameters(),
                                    lr=lr, eps=1e-08, betas=(args.beta1, args.beta2))
        return opt_vq, opt_disc
        
    def start_val_csv(self,args, epoch, fname_fmt="val_corr_e{epoch:04d}_{ts}.csv"):
        """
        每次验证开始时调用，返回 (writer, file_handle)。
        文件名包含 epoch 和时间戳，天然避免覆盖。
        """
        os.makedirs(args.log_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(args.log_dir, fname_fmt.format(epoch=epoch, ts=ts))
        f = open(path, "w", newline="")
        w = csv.writer(f)
        w.writerow(["epoch", "iter_in_epoch", "global_val_iter", "avg_corr", "rec_loss"])
        return w, f, path

    def train(self, args):
        os.makedirs(args.log_dir, exist_ok=True)
        writer = SummaryWriter(args.log_dir)

        csv_path = "/train_sub_slices_wide.csv"
        
        ds_pair = PairWindowDataset(csv_path,
                                cond_a="a",
                                cond_b="b",
                                k=12, row_idx=None, seed=123)
        print('ds_pair 长度是',len(ds_pair))
        train_loader = DataLoader(ds_pair, batch_size=args.batch_size, shuffle=True, num_workers=2)
        
        csv_path = "/test_sub_slices_wide.csv"
        
        ds_pair_ts = PairWindowDataset(csv_path,
                                cond_a="a",
                                cond_b="b",
                                k=12, row_idx=None, seed=123)
        print('ds_pair_ts 长度是',len(ds_pair_ts))
        test_loader = DataLoader(ds_pair_ts, batch_size=args.batch_size, shuffle=True, num_workers=2)
    
        steps_one_epoch = len(train_loader)
        total_steps=0
        val_iter=0
        
        # 2) 明确放置设备：只量化器在 GPU，其它都在 CPU
        self.vqgan.encoder.cpu()
        self.vqgan.decoder.cpu()
        self.vqgan.quant_conv.cpu()
        self.vqgan.post_quant_conv.cpu()

        # 视你的命名而定：常见是 codebook/quantizer 两个模块中的至少一个（或合在一起）
        self.vqgan.codebook.to(device=args.device)          # 如果有
        
        # ---- 冻结整个 VQGAN ----
        self.vqgan.eval()                         # 关闭 BN/Dropout 等
        self.vqgan.requires_grad_(False)          # 递归把所有参数的 requires_grad 设为 False
        # （可选）再显式确保 codebook 也为 False（稳妥起见）
        for p in self.vqgan.codebook.parameters():
            p.requires_grad = False
            
        shape_4d = (12, 4, 4, 4)
        L = 12 * 4 * 4 * 4   # 768
        width_a = width_b = 256
        layers_a = layers_b = 4
        heads_a = heads_b = 4
        embed_dim = 512
        code_dim = args.latent_dim
        model = DualViTCLIP_BLD(
            L=L,
            width_a=width_a, width_b=width_b,
            layers_a=layers_a, layers_b=layers_b,
            heads_a=heads_a, heads_b=heads_b,
            embed_dim=embed_dim,
            shape_4d=shape_4d,
            code_dim_a=code_dim, code_dim_b=code_dim
        ).to(args.device)
        
        # ====== Optimizer（最简洁：一个 AdamW）======
        use_amp = True
        optimizer = torch.optim.AdamW(
            model.parameters(),           # 只训练 CLIP 双塔
            lr=3e-4,                      # 常用起点
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.01
        )
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

        # ====== Train Loop（最小可跑）======
        total_step=0
        for epoch in range(args.epochs):
            model.train()
            pbar = tqdm(train_loader, desc=f"epoch {epoch+1}/{args.epochs}")
            avg_loss, avg_acc, n = 0.0, 0.0, 0
            
            if epoch==3:
                break

            for step, (xa, xb, meta) in enumerate(pbar):
                total_step+=1
                
                if step==3:
                    break
            
                # 1) indices 放到与 codebook 同设备（你上面 args.device）
                xa = xa.to(args.device, non_blocking=True)  # [B,k,D,H,W] long
                xb = xb.to(args.device, non_blocking=True)

                # 2) 冻结VQGAN，仅查表（无梯度）
                with torch.no_grad():
                    # indices_embed_*: [B, k, D, H, W, code_dim]
                    indices_embed_a = self.vqgan.codebook.embedding(xa)
                    indices_embed_b = self.vqgan.codebook.embedding(xb)
                    
                    B,k,D,H,W,C = indices_embed_a.shape
                    L = k*D*H*W
                    fa = indices_embed_a.view(B, L, C).mean(dim=1)  # [B, C]
                    fb = indices_embed_b.view(B, L, C).mean(dim=1)  # [B, C]
                    
                    # L2 归一化后做相似度矩阵
                    fa = F.normalize(fa.float(), dim=1)
                    fb = F.normalize(fb.float(), dim=1)

                    sim = fa @ fb.t()                          # [B, B], 余弦相似度
                    B = sim.size(0)
                    diag = sim.diag().mean().item()
                    off  = (sim.sum() - sim.diag().sum()).div(sim.numel() - B).item()

                    # 打印/记录
                    print(f"[codebook embed] diag={diag:.3f}, off={off:.3f}")

                # 3) TokLIP式 MLP：flatten -> width，得到 tokens [B, L, width]
                tokens_a = model.preprocess_indices_embed_a(indices_embed_a)
                tokens_b = model.preprocess_indices_embed_b(indices_embed_b)

                # 4) 前向 + 对比损失（对称 InfoNCE）
                with torch.cuda.amp.autocast(enabled=use_amp):
                    logits_ab, logits_ba, (za, zb) = model(tokens_a, tokens_b)  # [B,B]
                    loss, acc = contrastive_loss(logits_ab, logits_ba)
                    
                # logits_ab 是带温度的；拿归一化前的 cos 更稳定：
                cos = (za @ zb.t())                      # [-1,1]
                B = cos.size(0)
                off_mask = ~torch.eye(B, dtype=torch.bool, device=cos.device)
                off_penalty = (cos[off_mask]**2).mean()  # 惩罚非对角接近±1
                loss = loss + 0.2 * off_penalty         # 系数(0.01~0.1)自己调

                # 5) 反传与更新
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                writer.add_scalar(f'train/loss', loss.item(), total_step)
                writer.add_scalar(f'train/acc', acc, total_step)
                
                with torch.no_grad():
                    # 直接在 GPU 上算，避免来回拷贝；用 float() 防 AMP 混精度
                    s = (za @ zb.t()).float()                     # [B,B], 余弦相似度
                    B = s.size(0)
                    diag = s.diag().mean().item()
                    off  = (s.sum() - s.diag().sum()).div(s.numel() - B).item()
                    # 用 tqdm 的安全输出
                    pbar.write(f"[dbg] step={total_step} diag={diag:.3f} off={off:.3f} "
                            f"temp={model.logit_scale.exp().item():.2f}")
                    # 或者显示在进度条尾部
                    pbar.set_postfix(diag=f"{diag:.3f}", off=f"{off:.3f}",
                                    temp=f"{model.logit_scale.exp().item():.2f}")
                    
                    writer.add_scalar('dbg/mean_diag', diag, total_step)
                    writer.add_scalar('dbg/mean_off',  off,  total_step)

                
            if epoch%30==0:
                torch.save(model.state_dict(), os.path.join(args.checkd_dir, f"clip_epoch_{epoch}.pt"))
                
            # ---- epoch end: test ----
            test_loss, test_acc = evaluate_clip(model, self.vqgan, test_loader,
                                                device=args.device, use_amp=use_amp)
            print(f"[test] epoch {epoch+1}: loss={test_loss:.4f}, acc={test_acc*100:.2f}%, "
                f"temp={model.logit_scale.exp().item():.3f}")
            
            writer.add_scalar(f'val/test_loss', test_loss, total_step)
            writer.add_scalar(f'val/test_acc', test_acc, total_step)

                   

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="VQGAN")
    parser.add_argument('--latent-dim', type=int, default=256, help='Latent dimension n_z (default: 256)')
    # parser.add_argument('--latent-dim', type=int, default=64, help='Latent dimension n_z (default: 256)')
    parser.add_argument('--image-size', type=int, default=48, help='Image height and width (default: 256)')
    # parser.add_argument('--image-size', type=int, default=48, help='Image height and width (default: 256)')
    parser.add_argument('--num-codebook-vectors', type=int, default=512, help='Number of codebook vectors (default: 256)')
    parser.add_argument('--beta', type=float, default=0.25, help='Commitment loss scalar (default: 0.25)')
    parser.add_argument('--image-channels', type=int, default=1, help='Number of channels of images (default: 3)')
    parser.add_argument('--dataset-path', type=str, default='/data', help='Path to data (default: /data)')
    parser.add_argument('--device', type=str, default="cuda:7", help='Which device the training is on')
    parser.add_argument('--batch-size', type=int, default=12, help='Input batch size for training (default: 6)')
    parser.add_argument('--epochs', type=int, default=800, help='Number of epochs to train (default: 50)')
    parser.add_argument('--learning-rate', type=float, default=2.25e-05, help='Learning rate (default: 0.0002)')
    parser.add_argument('--beta1', type=float, default=0.5, help='Adam beta param (default: 0.0)')
    parser.add_argument('--beta2', type=float, default=0.9, help='Adam beta param (default: 0.999)')
    parser.add_argument('--disc-start', type=int, default=1000, help='When to start the discriminator (default: 0)')
    parser.add_argument('--disc-factor', type=float, default=1., help='')
    parser.add_argument('--l2-loss-factor', type=float, default=2., help='Weighting factor for reconstruction loss.')
    parser.add_argument('--perceptual-loss-factor', type=float, default=1., help='Weighting factor for perceptual loss.')
    parser.add_argument('--pearson-weight', type=float, default=0.25, help='Weighting pearson_weight.')
    parser.add_argument('--base_dir', type=str, default='/home/clip_out/', help='')
    parser.add_argument('--result_dir', type=str, default=None, help='')
    parser.add_argument('--checkd_dir', type=str, default=None, help='')
    parser.add_argument('--log_dir', type=str, default=None, help='')
    parser.add_argument('--expr_name', type=str, default='abcd3d', help='')
    
    parser.add_argument('--ckpt_path', type=str, default='/vqgan_epoch_30.pt', help='')

    args = parser.parse_args()
    
    args.channels_encoder=[128, 128, 256, 256, 512]
    args.channels_decoder=[128, 128, 256, 512]

    train_vqgan = TrainVQGAN(args)


