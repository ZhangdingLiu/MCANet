import torch
from torch import nn
from torchvision.models import efficientnet

class EfficientNetMultiLabel(nn.Module):
    def __init__(self, base_model: nn.Module, num_classes: int):
        super(EfficientNetMultiLabel, self).__init__()
        self.features = base_model.features
        self.avgpool = base_model.avgpool
        self.classifier = nn.Sequential(
            nn.Dropout(p=base_model.classifier[0].p, inplace=True),
            nn.Linear(base_model.classifier[1].in_features, num_classes),
        )
        self.loss_func = nn.BCEWithLogitsLoss()

    def forward(self, x: torch.Tensor, target: torch.Tensor = None):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        if target is not None:
            loss = self.loss_func(x, target)
            return x, loss
        else:
            return x

def efficientnet_b0_nocsra(pretrained: bool = True, num_classes: int = 1000):
    base_model = efficientnet.efficientnet_b0(pretrained=pretrained)
    return EfficientNetMultiLabel(base_model, num_classes)

def efficientnet_b1_nocsra(pretrained: bool = True, num_classes: int = 1000):
    base_model = efficientnet.efficientnet_b1(pretrained=pretrained)
    return EfficientNetMultiLabel(base_model, num_classes)

def efficientnet_b2_nocsra(pretrained: bool = True, num_classes: int = 1000):
    base_model = efficientnet.efficientnet_b2(pretrained=pretrained)
    return EfficientNetMultiLabel(base_model, num_classes)

def efficientnet_b3_nocsra(pretrained: bool = True, num_classes: int = 1000):
    base_model = efficientnet.efficientnet_b3(pretrained=pretrained)
    return EfficientNetMultiLabel(base_model, num_classes)

def efficientnet_b4_nocsra(pretrained: bool = True, num_classes: int = 1000):
    base_model = efficientnet.efficientnet_b4(pretrained=pretrained)
    return EfficientNetMultiLabel(base_model, num_classes)

def efficientnet_b5_nocsra(pretrained: bool = True, num_classes: int = 1000):
    base_model = efficientnet.efficientnet_b5(pretrained=pretrained)
    return EfficientNetMultiLabel(base_model, num_classes)

def efficientnet_b6_nocsra(pretrained: bool = True, num_classes: int = 1000):
    base_model = efficientnet.efficientnet_b6(pretrained=pretrained)
    return EfficientNetMultiLabel(base_model, num_classes)

def efficientnet_b7_nocsra(pretrained: bool = True, num_classes: int = 1000):
    base_model = efficientnet.efficientnet_b7(pretrained=pretrained)
    return EfficientNetMultiLabel(base_model, num_classes)
