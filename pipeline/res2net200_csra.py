
from .csra import CSRA, MHA

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from res2net.res2net import Res2Net, Bottle2neck, model_urls
import torch.nn.functional as F

class CustomRes2Net(Res2Net):
    def __init__(self, block, layers, baseWidth=26, scale=4, pretrained=False, **kwargs):
        super(CustomRes2Net, self).__init__(block, layers, baseWidth=baseWidth, scale=scale, **kwargs)

        # Load the pretrained model weights manually
        if pretrained:
            model_url = model_urls['res2net200_v1b_26w_4s']  # URL for the best Res2Net200 model
            state_dict = model_zoo.load_url(model_url)
            self.load_state_dict(state_dict)

    def forward(self, x):
        # Forward pass through Res2Net layers, without avgpool and view
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Return the feature map directly without pooling or flattening
        return x

class Res2Net200_Csra(nn.Module):
    def __init__(self, num_heads, lam, num_classes, input_dim=2048, pretrained=True):
        super(Res2Net200_Csra, self).__init__()
        # Use the custom Res2Net model with Res2Net200_v1b configuration
        self.res2net = CustomRes2Net(Bottle2neck, [3, 24, 36, 3], baseWidth=26, scale=4, pretrained=pretrained)

        # Replace the classifier part with MHA
        self.res2net.fc = MHA(num_heads, lam, input_dim, num_classes)

        # Binary cross-entropy loss function
        self.loss_func = F.binary_cross_entropy_with_logits

    def forward(self, x, target=None):
        # Perform forward pass through convolution and residual layers (conv1 to layer4), outputting the feature map
        x = self.res2net(x)

        # Classification using MHA, output shape (B, num_classes)
        x = self.res2net.fc(x)

        if target is not None:
            # If target is provided, compute the loss
            loss = self.loss_func(x, target, reduction="mean")
            return x, loss
        else:
            # Otherwise, return the predictions
            return x
