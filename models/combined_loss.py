import torch
import torch.nn as nn
import torch.nn.functional as F

class CombinedLoss(nn.Module):
    def __init__(self, ce_weight=None, dice_smooth=1e-5, dice_class_weights=None):
        super().__init__()
        self.dice_smooth = dice_smooth
        self.ce = nn.CrossEntropyLoss(weight=ce_weight)
        self.dice_class_weights = dice_class_weights  # 形如 [2.0, 1.0, 0.5]（肿瘤权重高）

    def forward(self, y_pred, y_true):
        # --- 交叉熵损失 ---
        loss_ce = self.ce(y_pred, y_true)

        # --- 多类别 Dice 损失 ---
        y_prob = F.softmax(y_pred, dim=1)  # [B, 3, H, W]
        y_true_onehot = F.one_hot(y_true, num_classes=3).permute(0, 3, 1, 2).float()  # [B, 3, H, W]

        intersection = torch.sum(y_prob * y_true_onehot, dim=(2, 3))  # [B, 3]
        union = torch.sum(y_prob + y_true_onehot, dim=(2, 3))         # 修正后的并集计算 [B, 3]
        dice = (2. * intersection + self.dice_smooth) / (union + self.dice_smooth)  # [B, 3]

        # 加权平均 Dice 系数
        if self.dice_class_weights is not None:
            weights = torch.tensor(self.dice_class_weights, device=dice.device)
            dice = dice * weights
        loss_dice = 1 - dice.mean()  # 加权平均

        # 总损失
        total_loss = loss_ce + loss_dice
        return total_loss