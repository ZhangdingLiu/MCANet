from torchvision.models import ResNet
from torchvision.models.resnet import Bottleneck, BasicBlock
import torch.utils.model_zoo as model_zoo
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

class ResNet152_Multilabel(ResNet):
    arch_settings = {
        18: (BasicBlock, (2, 2, 2, 2)),
        34: (BasicBlock, (3, 4, 6, 3)),
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3))
    }

    def __init__(self, num_classes, depth=152, input_dim=2048, cutmix=None):
        self.block, self.layers = self.arch_settings[depth]
        self.depth = depth
        super(ResNet152_Multilabel, self).__init__(self.block, self.layers)
        self.init_weights(pretrained=True, cutmix=cutmix)
        self.fc = nn.Linear(input_dim, num_classes)  # feature dim: 2048
        # self.classifier = input_dim, num_classes)
        self.loss_func = F.binary_cross_entropy_with_logits

    def backbone(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)  # after this, thinking whether should add this part of code to the forward part. because the cam attention is difficult to alter
        x = x.view(x.size(0), -1)  # (batch_size, 2048, 1, 1)  (batch_size, 2048);
        return x    # first dimension is the batch size and the second dimension is the number of features. or maybe this view operation should add to the forward function

    def forward_train(self, x, target):
        x = self.backbone(x)
        logit = self.fc(x)
        loss = self.loss_func(logit, target, reduction="mean")
        return logit, loss

    def forward_test(self, x):
        x = self.backbone(x)
        x = self.fc(x)
        return x

    def forward(self, x, target=None):
        if target is not None:
            return self.forward_train(x, target)
        else:
            return self.forward_test(x)

    def init_weights(self, pretrained=True, cutmix=None):
        if cutmix is not None:
            print("backbone params inited by CutMix pretrained model")
            state_dict = torch.load(cutmix)
        elif pretrained:
            print("backbone params inited by Pytorch official model")
            model_url = model_urls["resnet{}".format(self.depth)]
            state_dict = model_zoo.load_url(model_url)

        model_dict = self.state_dict()
        try:
            pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict}
            self.load_state_dict(pretrained_dict)
        except:
            logger = logging.getLogger()
            logger.info(
                "the keys in pretrained model is not equal to the keys in the ResNet you choose, trying to fix...")
            state_dict = self._keysFix(model_dict, state_dict)
            self.load_state_dict(state_dict)
        # remove the original 1000-class fc  , this is right

