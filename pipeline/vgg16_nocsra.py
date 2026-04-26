import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16
import torch.utils.model_zoo as model_zoo

class VGG16_Multilabel(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(VGG16_Multilabel, self).__init__()
        self.vgg = vgg16(pretrained=pretrained)
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
            x = torch.sigmoid(x)  # sigmoid
            return x

    # def init_weights(self, pretrained=True):
    #     if pretrained:
    #         print("Initializing weights from pre-trained model")
    #         model_url = model_urls['vgg16']
    #         state_dict = model_zoo.load_url(model_url)
    #         self.vgg.load_state_dict(state_dict)

