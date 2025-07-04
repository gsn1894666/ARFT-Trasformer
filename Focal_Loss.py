import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, gamma=0.5, alpha=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma  # Focusing parameter
        self.alpha = alpha  # Weighting factor for class imbalance (optional)
        if alpha is not None:
            if isinstance(alpha, (float, int)):
                self.alpha = torch.tensor([alpha, 1 - alpha])
            else:
                self.alpha = torch.tensor(alpha)
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Compute the cross-entropy loss (logits, not probabilities)
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')

        # Get the probabilities
        p = torch.exp(-ce_loss)  # p_t: probability of the true class

        # Compute focal loss
        focal_loss = (1 - p) ** self.gamma * ce_loss

        # Apply alpha weighting if provided
        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss

        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss