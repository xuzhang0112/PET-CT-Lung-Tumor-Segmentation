from typing import Optional, Sequence, Union
import warnings
import torch
import torch.nn.functional as F
from torch import Tensor
from monai.networks.utils import one_hot
from torch.nn.modules.loss import _Loss
from torch.nn import MSELoss, L1Loss, CrossEntropyLoss
from monai.losses.dice import DiceLoss
from monai.transforms import SobelGradients


device_ = "cpu"


class TotalLoss(_Loss):
    def __init__(self, device=device_, inclued_background=True, to_onehot_y=True, softmax=True, reduction: str = 'mean') -> None:
        super().__init__(reduction)
        self.l1 = L1Loss()
        self.cross_entropy = CrossEntropyLoss()
        self.dice_loss = DiceLoss(
            include_background=inclued_background, to_onehot_y=to_onehot_y, softmax=softmax)
        self.edge_loss = EdgeLoss(
            device=device, include_background=inclued_background, to_onehot_y=to_onehot_y, softmax=softmax)

    def l1_loss(self, input: torch.Tensor, target: torch.Tensor):
        # pytorch中l1_loss要求target和input同shape
        n_pred_ch, n_target_ch = input.shape[1], target.shape[1]
        target = one_hot(target, num_classes=n_pred_ch)
        return self.l1(input, target)

    def ce_loss(self, input: torch.Tensor, target: torch.Tensor):
        # pytorch中ce_loss要求target没有channel维
        target = torch.squeeze(target, dim=1)
        target = target.long()
        return self.cross_entropy(input, target)

    def forward(self, input, target):
        # compute_loss
        l1_loss = self.l1_loss(input, target)
        ce_loss = self.ce_loss(input, target)
        dice_loss = self.dice_loss(input, target)
        edge_loss = self.edge_loss(input, target)
        return l1_loss, ce_loss, dice_loss, edge_loss


class EdgeLoss(_Loss):
    def __init__(self, device=device_, include_background=True, softmax=False, sigmoid=False, to_onehot_y=False, reduction: str = 'mean') -> None:
        super().__init__(reduction=reduction)
        self.to_onehot_y = to_onehot_y
        self.include_background = include_background
        self.sigmoid = sigmoid
        self.softmax = softmax
        self.device = device
        self.mse_loss = MSELoss()

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        n_pred_ch = input.shape[1]
        if self.to_onehot_y:
            if n_pred_ch == 1:
                warnings.warn(
                    "single channel prediction, `to_onehot_y=True` ignored.")
            else:
                target = one_hot(target, num_classes=n_pred_ch)
        if self.sigmoid:
            input = torch.sigmoid(input)
        if self.softmax:
            if n_pred_ch == 1:
                warnings.warn(
                    "single channel prediction, `softmax=True` ignored.")
            else:
                input = torch.softmax(input, 1)
        tumor_edge1 = self.get_edge(torch.unsqueeze(input[:, 1], dim=1))
        background_edge1 = self.get_edge(torch.unsqueeze(input[:, 0], dim=1))
        edge1 = torch.concat([background_edge1, tumor_edge1], dim=1)
        tumor_edge2 = self.get_edge(torch.unsqueeze(target[:, 1], dim=1))
        background_edge2 = self.get_edge(torch.unsqueeze(target[:, 0], dim=1))
        edge2 = torch.concat([background_edge2, tumor_edge2], dim=1)
        edge_loss = self.mse_loss(edge1, edge2)
        return edge_loss

    def get_edge(self, feature_map):
        edge = Tensor([]).to(self.device)
        for i in range(feature_map.shape[0]):
            feature_map_ = feature_map[i]
            grads = SobelGradients()(feature_map_)
            edge = torch.concat(
                [edge, grads], dim=0)
        return edge


if __name__ == "__main__":
    x = torch.randn([8, 2, 512, 512])
    y = torch.unsqueeze(x.argmax(dim=1), dim=1)
    # y = torch.randn([8, 2, 512, 512])
    print(TotalLoss()(x, y))
