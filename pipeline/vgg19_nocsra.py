import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19

class VGG19_Multilabel(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(VGG19_Multilabel, self).__init__()
        self.vgg = vgg19(pretrained=pretrained)
        self.vgg.classifier[-1] = nn.Linear(4096, num_classes)
        self.loss_func = F.binary_cross_entropy_with_logits

    def forward(self, x, target=None):
        x = self.vgg.features(x)
        x = self.vgg.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.vgg.classifier(x)

        if target is not None:

            loss = self.loss_func(x, target)
            return x, loss
        else:
            # x = torch.sigmoid(x)
            return x

