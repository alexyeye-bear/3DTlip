# Tlip3D
Combine 3D VQ and contrastive learning.

## Training

### Stage 1: Train bidirectional Transformer

Train VQ part:

```bash
python train_vq.py
```

train transformer part: use any MLM modeling you like

For example, any bidirectional transformer, or auto-regressive transformer

Train transformer part:

```bash
python transformer.py
```

### Stage2 Train Clip part
```bash
python train_clip.py
```




