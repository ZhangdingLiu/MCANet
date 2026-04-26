import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v2

class MobileNetV2_Multilabel(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(MobileNetV2_Multilabel, self).__init__()
        # Load pre-trained MobileNetV2 model
        self.mobilenet = mobilenet_v2(pretrained=pretrained)
        # Replace the classifier part to suit the multi-label classification task
        self.mobilenet.classifier[1] = nn.Linear(self.mobilenet.classifier[1].in_features, num_classes)
        # Binary cross-entropy loss function
        self.loss_func = F.binary_cross_entropy_with_logits

    def forward(self, x, target=None):
        # Extract features and apply global average pooling
        x = self.mobilenet.features(x)  # Extract features
        x = x.mean([2, 3])              # Global average pooling

        x = self.mobilenet.classifier(x) # Classification

        if target is not None:
            # If target is provided, calculate the loss
            loss = self.loss_func(x, target)
            return x, loss
        else:
            # Otherwise, return the prediction
            return x
