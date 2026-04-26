import torch
import torch.nn as nn
# import torch.nn.functional as F
from res2net.res2net import res2net50
import torchvision.transforms as transforms

class Res2Net50_Multilabel(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(Res2Net50_Multilabel, self).__init__()
        self.res2net = res2net50(pretrained=pretrained)
        self.res2net.fc = nn.Linear(2048, num_classes)  # 2048
        self.loss_func = nn.BCEWithLogitsLoss()

    def forward(self, x, target=None):
        x = self.res2net(x)

        if target is not None:
            loss = self.loss_func(x, target)
            return x, loss
        else:
            return x
