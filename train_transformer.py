import os
import numpy as np
from tqdm import tqdm
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import utils as vutils
from transformer import VQGANTransformer
from utils_transf import load_data
from lr_schedule import WarmupLinearLRSchedule
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from PIL import Image

class TrainTransformer:
    def __init__(self, args):
        self.model = VQGANTransformer(args).to(device=args.device)
        self.optim = self.configure_optimizers()
        self.lr_schedule = WarmupLinearLRSchedule(
            optimizer=self.optim,
            init_lr=1e-6,
            peak_lr=args.learning_rate,
            end_lr=1e-7,
            warmup_epochs=5,
            epochs=args.epochs,
            current_step=args.start_from_epoch
        )
        
        # state_dict = torch.load(args.ckpt_path, map_location=args.device)
        # self.model.load_state_dict(state_dict)

        if args.start_from_epoch > 1:
            self.model.load_checkpoint(args.start_from_epoch)
            print(f"Loaded Transformer from epoch {args.start_from_epoch}.")
            
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        dir_path = os.path.join(args.base_dir, args.exr_name, timestamp)
        args.result_dir = os.path.join(dir_path, 'result')
        args.log_dir = os.path.join(dir_path, 'log')
        args.checkd_dir = os.path.join(dir_path, 'checks')
        args.output_dir = os.path.join(dir_path, 'output')

        self.train(args)
    
    def acc(self, target, inpainted, mask, sample_idx=2, save_path=None):
        """
        Compute accuracy only over masked positions, print masked preds & truths for a given sample,
        and optionally save the sample's mask as a PNG image (white=masked, black=unmasked).
        Args:
            target (Tensor): ground truth indices, shape [B, H, W]
            inpainted (Tensor): predicted indices, shape [B, H, W]
            mask (Tensor): boolean mask of same shape, True where evaluation should occur
            sample_idx (int): index of the sample to inspect
            save_path (str or None): if provided, path to save the mask image (.png)
        Returns:
            float: accuracy over masked positions
        """
        # Extract 2D mask for the specified sample
        sample_mask = mask[sample_idx].view(target.shape[1], target.shape[2])

        # Save mask image if a path is given
        if save_path:
            # Convert boolean mask to uint8 image (255 white, 0 black)
            mask_np = sample_mask.cpu().numpy().astype(np.uint8) * 255
            img = Image.fromarray(mask_np, mode='L')
            img.save(save_path)
            print(f"Saved sample {sample_idx} mask to {save_path}")

        # Print masked predictions and truths
        t_vals = target[sample_idx][mask[sample_idx]]
        ip_vals = inpainted[sample_idx][mask[sample_idx]]
        print(f"Sample {sample_idx} masked predictions: {ip_vals.tolist()}")
        print(f"Sample {sample_idx} masked truths: {t_vals.tolist()}")

        # Flatten tensors for overall accuracy
        target_flat = target.view(-1)
        inpainted_flat = inpainted.view(-1)
        mask_flat = mask.view(-1)
        total = mask_flat.sum().item()
        if total == 0:
            return 0.0

        # Compute accuracy
        matches = ((inpainted_flat == target_flat) & mask_flat).sum().item()
        accuracy = matches / total
        return accuracy

    def train(self, args):
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(args.log_dir, exist_ok=True)
        os.makedirs(args.checkd_dir, exist_ok=True)
        writer = SummaryWriter(args.log_dir)
        
        train_dataset = load_data(args,mode='train')
        val_dataset = load_data(args,mode='val')
        
        len_train_dataset = len(train_dataset)
        total_steps=0
        val_iter=0
        val_epoch_best=200000
        for epoch in range(args.start_from_epoch, args.epochs+1):
            # if epoch==1:
            #     break
            print(f"Epoch {epoch}:")
            with tqdm(range(len(train_dataset))) as pbar:
                self.lr_schedule.step()
                for i, (indices,sub_path) in zip(pbar, train_dataset):
                    # if i==1:
                    #     break
                    indices = indices.to(device=args.device)
                    logits, target = self.model(indices)
                    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1))
                    loss.backward()
                    if total_steps % args.accum_grad == 0:
                        self.optim.step()
                        self.optim.zero_grad()
                    total_steps += 1
                    pbar.set_postfix(Transformer_Loss=np.round(loss.cpu().detach().numpy().item(), 4))
                    pbar.update(0)
                    writer.add_scalar('metric/train Cross Entropy Loss', np.round(loss.cpu().detach().numpy().item(), 4),total_steps)
                
            if epoch % 3==0:
                val_epoch_sum=0
                self.model.transformer.eval()
                with torch.no_grad():
                    tr_acc=[]
                    for i, (indices,sub_path) in enumerate(val_dataset):
                        # if i==2:
                        #     break
                        val_iter = val_iter + 1
                        indices = indices.to(device=args.device)
                        # logits, target = self.model(indices)
                        # loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1))
                        # val_epoch_sum=val_epoch_sum+loss.cpu().detach().numpy().item()
                        
                        # 固定抽取一列置0，等价于416个时间点中的16个时间点
                        inpainted_ind,time_mask=self.model.sample_inpaint_timemask(indices)
                        # print('inpainted_ind',inpainted_ind.shape)
                        # print('time_mask',time_mask.shape)

                        one_b_acc=self.acc(indices,inpainted_ind,time_mask,2)
                        tr_acc.append(one_b_acc)
                        writer.add_scalar('metric/val_accuracy', one_b_acc,val_iter)
                        writer.add_scalar('metric/val', np.round(loss.cpu().detach().numpy().item()),val_iter)
                    total_acc=np.nanmean(np.array(tr_acc))
                    print('tr_acc',tr_acc)
                    print(f"val Overall accuracy: {total_acc*100:.2f}%")
                
                self.model.transformer.train()
                if val_epoch_best > val_epoch_sum:
                    val_epoch_best = val_epoch_sum
                    torch.save(self.model.state_dict(), os.path.join(args.checkd_dir, f"transformer_val_best.pt"))
                
            if epoch % args.ckpt_interval == 0:
                torch.save(self.model.state_dict(), os.path.join(args.checkd_dir, f"transformer_epoch_{epoch}.pt"))

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.transformer.parameters(), lr=1e-4, betas=(0.9, 0.96), weight_decay=4.5e-2)
        return optimizer

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="VQGAN")
    parser.add_argument('--run-name', type=str, default=None)
    parser.add_argument('--latent-dim', type=int, default=32, help='Latent dimension n_z.')
    parser.add_argument('--image-size', type=int, default=416, help='Image height and width.)')
    parser.add_argument('--num-codebook-vectors', type=int, default=256, help='Number of codebook vectors.')
    parser.add_argument('--beta', type=float, default=0.25, help='Commitment loss scalar.')
    parser.add_argument('--image-channels', type=int, default=1, help='Number of channels of images.')
    parser.add_argument('--dataset-path', type=str, default='./data', help='Path to data.')
    parser.add_argument('--checkpoint-path', type=str, default='./checkpoints/last_ckpt.pt', help='Path to checkpoint.')
    parser.add_argument('--device', type=str, default="cuda:4", help='Which device the training is on.')
    parser.add_argument('--batch-size', type=int, default=10, help='Batch size for training.')
    parser.add_argument('--accum-grad', type=int, default=10, help='Number for gradient accumulation.')
    parser.add_argument('--epochs', type=int, default=300, help='Number of epochs to train.')
    parser.add_argument('--start-from-epoch', type=int, default=0, help='Number of epochs to train.')
    parser.add_argument('--ckpt-interval', type=int, default=20, help='Number of epochs to train.')
    parser.add_argument('--learning-rate', type=float, default=1e-4, help='Learning rate.')

    parser.add_argument('--sos-token', type=int, default=677, help='Start of Sentence token.')

    parser.add_argument('--n-layers', type=int, default=16, help='Number of layers of transformer.')
    parser.add_argument('--dim', type=int, default=256, help='Dimension of transformer.')
    parser.add_argument('--hidden-dim', type=int, default=1024, help='Dimension of transformer.')
    parser.add_argument('--num-image-tokens', type=int, default=676, help='Number of image tokens.')
    
    parser.add_argument('--base_dir', type=str, default='/home/ts_git_out/', help='')
    
    parser.add_argument('--vitvq_ckpt_path', type=str, default='/home/vqgan_epoch_290.pt', help='')

    args = parser.parse_args()
    args.exr_name = "transf_2d"
    
    args.vq_latentdim = 256
    args.vq_image_size = 416
    args.vq_patch_size=16
    args.vq_transformer_layers = 8
    args.vq_transformer_heads = 8
    args.vq_transformer_dim_head = 32
    args.vq_transformer_mlp_mult = 4

    train_transformer = TrainTransformer(args)
    
