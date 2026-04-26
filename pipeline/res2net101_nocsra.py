import torch
import torch.nn as nn
import torch.nn.functional as F
from res2net.res2net import res2net101_26w_4s
import torchvision.transforms as transforms

class Res2Net101_Multilabel(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(Res2Net101_Multilabel, self).__init__()
        self.res2net = res2net101_26w_4s(pretrained=pretrained)
        self.res2net.fc = nn.Linear(2048, num_classes)
        self.loss_func = F.binary_cross_entropy_with_logits

    def extractfeature(self, x):
        x = self.res2net.conv1(x)
        x = self.res2net.bn1(x)
        x = self.res2net.relu(x)
        x = self.res2net.maxpool(x)
        x = self.res2net.layer1(x)
        x = self.res2net.layer2(x)
        x = self.res2net.layer3(x)
        feature_map = self.res2net.layer4(x)  # layer4
        return feature_map  # feature map  (batch_size, 2048, 7, 7)

    def forward(self, x, target=None):
        x = self.res2net(x)

        if target is not None:
            loss = self.loss_func(x, target)
            return x, loss
        else:
            return x

