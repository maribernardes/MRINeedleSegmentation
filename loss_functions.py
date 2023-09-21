import torch
from monai.losses import TverskyLoss, FocalLoss

class UnifiedFocalLoss(torch.nn.Module):
    def __init__(self, delta=0.6, gamma=0.5, lambda_=0.5, to_onehot_y=False, softmax=False):
        super(UnifiedFocalLoss, self).__init__()
        self.delta = delta
        self.gamma = gamma
        self.lambda_ = lambda_
        self.to_onehot_y = to_onehot_y
        self.softmax = softmax
        self.tversky = TverskyLoss(alpha = self.delta, beta = (1-self.delta),
                                   reduction="mean", to_onehot_y=self.to_onehot_y, softmax=self.softmax)
        self.focal = FocalLoss(gamma = 1-self.gamma, 
                               reduction="mean", to_onehot_y=self.to_onehot_y, use_softmax=self.softmax)
    def forward(self, x, y):
        return self.lambda_ * self.delta * self.focal(x, y) + (1 - self.lambda_) * self.tversky(x, y)