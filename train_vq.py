import os
import argparse
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import utils as vutils
from discriminator3d import Discriminator3D
import lpips
from vqgan3d import VQGAN3D
from torch.utils.tensorboard import SummaryWriter
import torchvision
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
import csv
from dataloaders.custom_loader import weights_init,RandomSliceFromCSV

class TrainVQGAN:
    def __init__(self, args):
        self.vqgan = VQGAN3D(args)
        
        # state_dict = torch.load(args.ckpt_path, map_location=args.device)
        # self.vqgan.load_state_dict(state_dict)
        
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
        
        bmask_np=np.load('/home/mask.npy')
        
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
    
    def _masked_corr_per_sample(self, gen: torch.Tensor, real: torch.Tensor, eps: float = 1e-6):
        """
        返回 (corr_vec, valid_mask)：
        corr_vec   [B]：每个样本在掩膜内的皮尔逊相关
        valid_mask [B]：方差非零的样本
        """
        B = gen.size(0)

        # 展平后按掩膜取子集
        mask_flat = self.bmask_t.view(-1)
        if mask_flat.device != gen.device:
            mask_flat = mask_flat.to(gen.device, non_blocking=True)

        X = gen.view(B, -1)[:, mask_flat]   # [B, K]
        Y = real.view(B, -1)[:, mask_flat]  # [B, K]

        # 去均值
        Xc = X - X.mean(dim=1, keepdim=True)
        Yc = Y - Y.mean(dim=1, keepdim=True)

        # 皮尔逊相关：中心化后的余弦相似度
        num   = (Xc * Yc).sum(dim=1)                # [B]
        xnorm = torch.linalg.norm(Xc, dim=1)        # [B]
        ynorm = torch.linalg.norm(Yc, dim=1)        # [B]
        denom = xnorm * ynorm                       # [B]

        valid = denom > eps
        corr  = torch.zeros(B, device=gen.device, dtype=gen.dtype)
        corr[valid] = num[valid] / denom[valid]
        return corr, valid

    def Pearson_corr_whole(self, gen: torch.Tensor, real: torch.Tensor, eps: float = 1e-6):
        """返回 batch 内有效样本的皮尔逊相关均值（标量）。"""
        corr, valid = self._masked_corr_per_sample(gen, real, eps)
        if valid.any():
            return corr[valid].mean()
        else:
            return torch.zeros((), device=gen.device, dtype=gen.dtype)


    def Pearson_loss_whole(self, gen: torch.Tensor, real: torch.Tensor,
                        eps: float = 1e-6, rate: float = 1.0):
        """
        定义为平均的 (1 - corr)^2（仅对有效样本求平均），再乘以 rate。
        """
        corr, valid = self._masked_corr_per_sample(gen, real, eps)
        if valid.any():
            return ((1.0 - corr[valid]) ** 2).mean() * rate
        else:
            return torch.zeros((), device=gen.device, dtype=gen.dtype)
        
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

        csv_path = "split1.csv"
        data_root = "datatrain"

        ds_tr = RandomSliceFromCSV(
            csv_path,
            condition_cols=["a1", "b1"],
            data_root=data_root,
            run="run-01",
            split='train',
            suffix=".npy",
            seed=1234
        )
        train_loader = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True, num_workers=2)
        
        print('ds 长度是',len(ds_tr))
        
        csv_path = "split1.csv"
        data_root = "data_test"

        ds_ts = RandomSliceFromCSV(
            csv_path,
            condition_cols=["a1", "b1"],
            data_root=data_root,
            run="run-01",
            split='test',
            suffix=".npy",
            seed=1234
        )
        test_loader = DataLoader(ds_ts, batch_size=args.batch_size, shuffle=True, num_workers=2)
        
        print('ds_ts 长度是',len(ds_ts))
        
        steps_one_epoch = len(train_loader)
        total_steps=0
        val_iter=0
        for epoch in range(args.epochs):
            # if epoch==1:
            #     break
            with tqdm(range(len(train_loader))) as pbar:
                for i, (imgs,p) in zip(pbar, train_loader):
                    # if i==1:
                    #     break
                    imgs = imgs.to(device=args.device)
                    decoded_images, _, q_loss = self.vqgan(imgs)
                    # print('decoded_images shape is',decoded_images.shape)

                    disc_real = self.discriminator(imgs)
                    disc_fake = self.discriminator(decoded_images)

                    disc_factor = self.vqgan.adopt_weight(args.disc_factor, epoch * steps_one_epoch + i,
                                                          threshold=args.disc_start)
                    
                    #---------------------loss--------------------------------
                    N, C, D, H, W = imgs.shape 
                    imgs_per = imgs.permute(0, 3, 1, 2, 4).contiguous()
                    imgs_2d_h = imgs_per.view(N * H, C, D, W)
                    decoded_images_per = decoded_images.permute(0, 3, 1, 2, 4).contiguous()
                    decoded_images_2d_h = decoded_images_per.view(N * H, C, D, W)
                    
                    perceptual_loss = self.perceptual_loss(imgs_2d_h, decoded_images_2d_h)
                    # print("Perceptual Loss shape:", perceptual_loss.shape)
                    rec_loss = torch.abs(imgs - decoded_images)
                    rec_loss = rec_loss.mean(dim=[1, 2, 3, 4], keepdim=True)  # [B, 1, 1, 1, 1]
                    rec_loss = rec_loss.view(-1, 1)                      # reshape to [B, 1]
                    # print("rec_loss shape:", rec_loss.shape)
                    
                    per_loss_mean = perceptual_loss.mean()   # 标量
                    rec_loss_mean = rec_loss.mean()          # 标量（rec_loss 目前是 [B,1]）
                    
                    pearson_loss = self.Pearson_loss_whole(decoded_images,imgs)
                    # print("pearson_loss:", pearson_loss)
                    
                    nll_losss = args.perceptual_loss_factor * per_loss_mean + args.l2_loss_factor * rec_loss_mean + pearson_loss*args.pearson_weight
                    
                    g_loss = -torch.mean(disc_fake)

                    λ = self.vqgan.calculate_lambda(nll_losss, g_loss)
                    loss_vq = nll_losss + q_loss + disc_factor * λ * g_loss

                    d_loss_real = torch.mean(F.relu(1. - disc_real))
                    d_loss_fake = torch.mean(F.relu(1. + disc_fake))
                    loss_gan = disc_factor * .5 * (d_loss_real + d_loss_fake)

                    self.opt_vq.zero_grad()
                    loss_vq.backward(retain_graph=True)

                    self.opt_disc.zero_grad()
                    loss_gan.backward()

                    self.opt_vq.step()
                    self.opt_disc.step()
                    
                    total_steps = total_steps + 1

                    if total_steps%400==0:
                        writer.add_scalar(f'train/perceptual_loss', per_loss_mean.item(), total_steps)
                        writer.add_scalar(f'train/rec_loss', rec_loss_mean.item(), total_steps)
                        writer.add_scalar(f'train/nll_losss', nll_losss.item(), total_steps)
                        writer.add_scalar(f'train/loss_gan', loss_gan.item(), total_steps)
                        writer.add_scalar('metric/train pearson_loss', pearson_loss.item(), total_steps)

                    pbar.set_postfix(VQ_Loss=np.round(loss_vq.cpu().detach().numpy().item(), 5),
                                     GAN_Loss=np.round(loss_gan.cpu().detach().numpy().item(), 3))
                    pbar.update(0)
                
                if epoch % 3==0:
                    torch.save(self.vqgan.state_dict(), os.path.join(args.checkd_dir, f"vqgan_epoch_{epoch}.pt"))

                # total_iter = total_iter + steps_one_epoch
                if epoch % 1==0:
                    # —— 验证开始处：
                    corr_writer, corr_f, corr_path = self.start_val_csv(args, epoch)
                    print(f"[VAL] writing corr to: {corr_path}")

                    self.vqgan.eval()
                    
                    val_show_interval=4000
                    with torch.no_grad():
                        for i, (imgs,p) in enumerate(test_loader):
                            val_iter = val_iter + 1
                            # if i==5:
                            #     break
                            imgs = imgs.to(device=args.device)
                            decoded_images, _, _ = self.vqgan(imgs)

                            # 假设你有一个 4D 张量（NCHW格式）
                            
                            if val_iter%val_show_interval==0:
                                images = imgs[:,0,:,:,30].unsqueeze(1)
                                grid_image = torchvision.utils.make_grid(images,nrow=4,normalize=True,scale_each=True)      # 对每张子图各自归一化)
                                writer.add_image('images', grid_image, val_iter)  # 添加图像数据到 TensorBoard
                                images = decoded_images[:,0,:,:,30].unsqueeze(1)
                                grid_image = torchvision.utils.make_grid(images,nrow=4,normalize=True,scale_each=True) 
                                writer.add_image('pred', grid_image, val_iter)  # 添加图像数据到 TensorBoard
                            
                            rec_loss = torch.abs(imgs - decoded_images)
                            rec_loss_scalar = rec_loss.mean()
                            
                            y_true,y_pred=imgs.detach().cpu().squeeze(),decoded_images.detach().cpu().squeeze()
                            avg_corr = self.Pearson_corr_whole(y_true, y_pred)
                            
                            if val_iter%val_show_interval==0:
                                print(f"Average Pearson correlation: {avg_corr:.4f}")
                            
                            writer.add_scalar(f'val/avg_corr', avg_corr.item(), val_iter)
                            
                            writer.add_scalar(f'val/rec_loss', rec_loss_scalar.item(), val_iter)
                            
                            # avg_corr: 标量张量；rec_loss_scalar: 标量张量

                            corr_writer.writerow([
                                epoch,
                                i,                 # 本 epoch 内第 i 个 iter
                                val_iter,          # 全局验证步
                                float(avg_corr.item()),
                                float(rec_loss_scalar.item()),
                            ])
                            corr_f.flush()         
                            
                            if i % val_show_interval == 0:
                                with torch.no_grad():
                                    B, C, D, H, W = imgs.shape
                                    d = D // 2

                                    def to01(x):                           
                                        return ((x + 1) * 0.5).clamp(0, 1)

                                    in2d   = imgs[:, :, d, :, :]           # [B,1,H,W]
                                    rec2d  = decoded_images[:, :, d, :, :] # [B,1,H,W]

                                    # print("bmask shape:", bmask.shape)

                                    # grid3 = torch.cat([to01(in2d), to01(rec2d), m2d], dim=0)  # [3B,1,H,W]
                                    grid3 = torch.cat([to01(in2d), to01(rec2d)], dim=0)  # [3B,1,H,W]
                                    vutils.save_image(grid3, os.path.join(args.result_dir, f"{epoch}_{i}.jpg"), nrow=B)

                    corr_f.close()
                    
                    self.vqgan.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="VQGAN")
    parser.add_argument('--latent-dim', type=int, default=256, help='Latent dimension n_z (default: 256)')
    parser.add_argument('--image-size', type=int, default=48, help='Image height and width (default: 256)')
    parser.add_argument('--num-codebook-vectors', type=int, default=512, help='Number of codebook vectors (default: 256)')
    parser.add_argument('--beta', type=float, default=0.25, help='Commitment loss scalar (default: 0.25)')
    parser.add_argument('--image-channels', type=int, default=1, help='Number of channels of images (default: 3)')
    parser.add_argument('--dataset-path', type=str, default='/data', help='Path to data (default: /data)')
    parser.add_argument('--device', type=str, default="cuda:2", help='Which device the training is on')
    parser.add_argument('--batch-size', type=int, default=32, help='Input batch size for training (default: 6)')
    parser.add_argument('--epochs', type=int, default=800, help='Number of epochs to train (default: 50)')
    parser.add_argument('--learning-rate', type=float, default=2.25e-05, help='Learning rate (default: 0.0002)')
    parser.add_argument('--beta1', type=float, default=0.5, help='Adam beta param (default: 0.0)')
    parser.add_argument('--beta2', type=float, default=0.9, help='Adam beta param (default: 0.999)')
    parser.add_argument('--disc-start', type=int, default=1000, help='When to start the discriminator (default: 0)')
    parser.add_argument('--disc-factor', type=float, default=1., help='')
    parser.add_argument('--l2-loss-factor', type=float, default=2., help='Weighting factor for reconstruction loss.')
    parser.add_argument('--perceptual-loss-factor', type=float, default=1., help='Weighting factor for perceptual loss.')
    parser.add_argument('--pearson-weight', type=float, default=0.25, help='Weighting pearson_weight.')
    parser.add_argument('--base_dir', type=str, default='/home/vq/', help='')
    parser.add_argument('--result_dir', type=str, default=None, help='')
    parser.add_argument('--checkd_dir', type=str, default=None, help='')
    parser.add_argument('--log_dir', type=str, default=None, help='')
    parser.add_argument('--expr_name', type=str, default='vq3d', help='')
    
    # parser.add_argument('--ckpt_path', type=str, default='vqgan_epoch_230.pt', help='')

    args = parser.parse_args()
    # args.channels_encoder=[128, 128, 256, 256, 512]
    # args.channels_decoder=[128, 128, 256, 512]

    train_vqgan = TrainVQGAN(args)


