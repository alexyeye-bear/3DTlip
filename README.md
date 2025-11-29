# 3DTlip
combine 3DVQ and contrastive learning

## Training

### Stage1 Train bi-directional transformer
train vq part
```bash
python train_vq.py


train transformer part: use any MLM modeling you like

### Stage2 Train Clip part
```bash
python train_clip.py






