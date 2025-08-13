import torch
import torch.nn as nn
import torch.nn.functional as F

class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.1, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            target = target.to(torch.int64)
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))




import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # 计算交叉熵损失
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        # 获取每个样本的概率
        pt = torch.exp(-ce_loss)
        # 计算 Focal Loss
        focal_loss = self.alpha * ((1 - pt) ** self.gamma) * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        # 如果是多分类任务，使用 softmax 处理预测
        inputs = torch.softmax(inputs, dim=1)
        targets_one_hot = nn.functional.one_hot(targets, num_classes=inputs.size(1)).permute(0, 3, 1, 2).float()
        
        # 计算 Dice 系数
        intersection = (inputs * targets_one_hot).sum(dim=(2, 3))
        union = inputs.sum(dim=(2, 3)) + targets_one_hot.sum(dim=(2, 3))
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        
        return 1.0 - dice.mean()


class BalancedLoss(nn.Module):
    def __init__(self, class_weights, alpha=1.0, gamma=2.0, lambda_ce=0.5):
        super(BalancedLoss, self).__init__()
        self.class_weights = class_weights
        self.focal_loss = FocalLoss(alpha, gamma)
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)
        self.lambda_ce = lambda_ce

    def forward(self, inputs, targets):
        targets = targets.long()
        focal_loss = self.focal_loss(inputs, targets)
        ce_loss = self.ce_loss(inputs, targets)
        return focal_loss + self.lambda_ce * ce_loss


