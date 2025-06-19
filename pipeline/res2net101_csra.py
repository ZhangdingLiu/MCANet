from .csra import CSRA, MHA

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from res2net.res2net import Res2Net, Bottle2neck, model_urls
import torch.nn.functional as F

class CustomRes2Net(Res2Net):
    def __init__(self, block, layers, baseWidth=26, scale=4, pretrained=False, **kwargs):
        super(CustomRes2Net, self).__init__(block, layers, baseWidth=baseWidth, scale=scale, **kwargs)

        if pretrained:
            model_url = model_urls['res2net101_26w_4s']
            state_dict = model_zoo.load_url(model_url)
            self.load_state_dict(state_dict)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x


class Res2Net101_Csra(nn.Module):
    def __init__(self, num_heads, lam, num_classes, input_dim=2048, pretrained=True):
        super(Res2Net101_Csra, self).__init__()

        self.res2net = CustomRes2Net(Bottle2neck, [3, 4, 23, 3], baseWidth=26, scale=4, pretrained=pretrained)

        self.res2net.fc = MHA(num_heads, lam, input_dim, num_classes)

        self.loss_func = F.binary_cross_entropy_with_logits

    def forward(self, x, target=None):
        x = self.res2net(x)

        x = self.res2net.fc(x)

        if target is not None:

            loss = self.loss_func(x, target, reduction="mean")
            return x, loss
        else:

            return x
