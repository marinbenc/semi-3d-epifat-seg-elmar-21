import torch.nn as nn
import torch

class DiceLoss(nn.Module):

    def __init__(self, weight = torch.Tensor([0.7, 0.3])):
        super(DiceLoss, self).__init__()
        self.smooth = 1.0
        self.weight = weight

    def forward(self, y_pred, y_true):
        assert y_pred.size() == y_true.size()

        dscs = torch.zeros(y_pred.shape[1])

        for i in range(y_pred.shape[1]):
            y_pred_ch = y_pred[:, i].contiguous().view(-1)
            y_true_ch = y_true[:, i].contiguous().view(-1)
            intersection = (y_pred_ch * y_true_ch).sum()
            dscs[i] = (2. * intersection + self.smooth) / (
                y_pred_ch.sum() + y_true_ch.sum() + self.smooth
            )

        # weighted average of dices
        return 1. - torch.sum(dscs * self.weight)