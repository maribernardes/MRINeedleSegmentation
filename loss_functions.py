import torch
from monai.losses import TverskyLoss, FocalLoss

class UnifiedFocalLoss(torch.nn.Module):
    def __init__(self, n_classes, delta = 0.6, gamma = 0.5, lambda_ = 0.5):
        super(UnifiedFocalLoss, self).__init__()
        self.N = n_classes
        self.delta = delta
        self.gamma = gamma
        self.lambda_ = lambda_
        self.tversky = TverskyLoss(include_background=True, alpha = self.delta, beta = (1-self.delta),
                                   reduction="mean", to_onehot_y=True, softmax=True)
        self.focal = FocalLoss(include_background=True, gamma = 1-self.gamma, reduction="mean", to_onehot_y=True, use_softmax=True)
    def forward(self, x, y):
        return self.lambda_ * self.delta * self.focal(x, y) + (1 - self.lambda_) * self.tversky(x, y)