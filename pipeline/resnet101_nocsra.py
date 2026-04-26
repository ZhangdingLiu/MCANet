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

# ResNet模型的例子
# 假设我们有一个标凈的ResNet模型，它包含多个卷积层（作为特征提取器，即backbone），
# 以及一个全连接层作为分类器（classifier）。在ResNet中，分类器通常是名为fc的单个全连接层。
# 如果我们遵循标准的ResNet实现，全连接层可能没有直接包含“classifier”这个名称，
# 因此上述代码可能需要调整以适应实际情况。
# 例如，如果我们知道全连接层的确切名称（在ResNet中是fc），代码应该修改为：

class ResNet101_Multilabel(ResNet):
    arch_settings = {
        18: (BasicBlock, (2, 2, 2, 2)),
        34: (BasicBlock, (3, 4, 6, 3)),
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3))
    }

    def __init__(self, num_classes, depth=101, input_dim=2048, cutmix=None):
        self.block, self.layers = self.arch_settings[depth]
        self.depth = depth
        super(ResNet101_Multilabel, self).__init__(self.block, self.layers)
        self.init_weights(pretrained=True, cutmix=cutmix)
        # 替换最后一层以适应多标签分类
        self.fc = nn.Linear(input_dim, num_classes)  # ResNet101的特征维度是2048
        self.loss_func = F.binary_cross_entropy_with_logits  # 二元交叉熵损失

    def extractfeature(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)  #
        # 不执行avgpool，直接返回最后一个卷积层的输出 (batch_size, 2048, 7, 7)
        return x

    def backbone(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)  # 7×7×2048 （input 224*224）
        x = self.avgpool(x)    # 1×1*2048 or 2048  我增加的 after this, thinking whether should add this part of code to the forward part. because the cam attention is difficult to alter
        x = x.view(x.size(0), -1)  #将(batch_size, 2048, 1, 1) 转换为 (batch_size, 2048);
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

    def forward(self, x, target=None):  # 根据是否提供了目标值，决定调用训练还是测试方法。
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

        # 如果预训练权重中的键与当前模型不匹配，则尝试修复键并加载权重。
        # 原始的全连接层（针对ImageNet 1000类任务）被一个空的序列替换掉了，因为后面会使用不同的分类器。
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




