import torch
import torch.nn as nn
from res2net.res2net import Res2Net, Bottle2neck
import torch.nn.functional as F
from .csra import CSRA, MHA

class CustomRes2Net152(Res2Net):
    def __init__(self, block, layers, baseWidth=26, scale=4, pretrained=False, weight_path=None, **kwargs):
        super(CustomRes2Net152, self).__init__(block, layers, baseWidth=baseWidth, scale=scale, **kwargs)

        # Load weights from a local file if pretrained is True and weight_path is provided
        if pretrained and weight_path:
            state_dict = torch.load(weight_path)
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

class Res2Net152_Csra(nn.Module):
    def __init__(self, num_heads, lam, num_classes, input_dim=2048, pretrained=True, weight_path=None):
        super(Res2Net152_Csra, self).__init__()
        # Use Res2Net-152 with custom weight loading
        self.res2net = CustomRes2Net152(Bottle2neck, [3, 8, 36, 3], baseWidth=26, scale=4, pretrained=pretrained, weight_path=weight_path)

        # Multi-label classification with MHA
        self.res2net.fc = MHA(num_heads, lam, input_dim, num_classes)

        # Binary cross-entropy loss for multi-label classification
        self.loss_func = F.binary_cross_entropy_with_logits

    def forward(self, x, target=None):
        x = self.res2net(x)  # Forward pass through Res2Net backbone
        x = self.res2net.fc(x)  # Classification with MHA

        if target is not None:
            # If target is provided, compute the loss
            loss = self.loss_func(x, target, reduction="mean")
            return x, loss
        else:
            return x
