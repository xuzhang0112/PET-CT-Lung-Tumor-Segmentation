import torch
from net import YNet, XNet, SelfAttentionXNet, CrossAttentionXNet
from monai.metrics import DiceMetric
from monai.utils.misc import set_determinism
from monai.data import CacheDataset, DataLoader
from dataset import train_files, train_transforms, val_files, val_transforms
import os
from loss import TotalLoss

num_epochs = 110
initial_lr = 2e-4
set_determinism()
# save_dir
save_dir = "/data/xuzhang/LungTumorSegmentation/L1+CE+Dice+SelfAttention+Edge"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
# device
device = torch.device("cuda:5")
# model
model = SelfAttentionXNet(1, 2).to(device)
# dataset
train_ds = CacheDataset(
    data=train_files, transform=train_transforms, cache_rate=1.0, num_workers=1)
train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=1)
val_ds = CacheDataset(
    data=val_files, transform=val_transforms, cache_rate=1.0, num_workers=1)
val_loader = DataLoader(val_ds, batch_size=1, num_workers=1)
# loss
l1 = 1
dice = 1
edge = 1000
loss_fn = TotalLoss(device)
# optimizer and scheduler
optimizer = torch.optim.Adam(
    model.parameters(), initial_lr, betas=(0.9, 0.99), weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=num_epochs)
# metric
pet_metric = DiceMetric(include_background=False, reduction="mean")
ct_metric = DiceMetric(include_background=False, reduction="mean")
