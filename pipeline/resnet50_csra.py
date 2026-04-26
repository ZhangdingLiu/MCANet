from torchvision.models import ResNet
from torchvision.models.resnet import Bottleneck, BasicBlock
from .csra import CSRA, MHA
import torch.utils.model_zoo as model_zoo
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

# 这段代码定义了一个名为 ResNet_CSRA 的类，它是对PyTorch库中提供的预训练ResNet模型的扩展，
# 增加了一个名为CSRA（ Relational Attention）的分类器。这个新模型旨在利用注意力机制
# （在这种情况下是多头注意力，或MHA）来改进分类性能。代码中包含了模型的构建、权重初始化、训练和测试的前向传播。
# 以下是详细的代码解释：

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

# ResNet_CSRA 类
# 继承自 torchvision.models.ResNet 类
# arch_settings：一个字典，将不同深度的ResNet模型映射到其相应的块类型和层结构上。
# __init__ 方法：初始化 ResNet_CSRA 对象。
#   num_heads：多头注意力机制中头的数量。
#   lam：可能是用于注意力机制中的某个正则化或缩放参数。
#   num_classes：模型应该分类的目标类别数量。
#   depth：指定ResNet的深度版本（例如ResNet50或ResNet101）。
#   input_dim：传入分类器的特征维数。
#   cutmix：一种数据增强技术，用于模型的预训练。
class ResNet50_CSRA(ResNet):
    arch_settings = {
        18: (BasicBlock, (2, 2, 2, 2)),
        34: (BasicBlock, (3, 4, 6, 3)),
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3))
    }

    def __init__(self, num_heads, lam, num_classes, depth=50, input_dim=2048, cutmix=None):
        self.block, self.layers = self.arch_settings[depth]
        self.depth = depth
        super(ResNet50_CSRA, self).__init__(self.block, self.layers)
        # 调用 ResNet_CSRA 类的父类 ResNet 的构造函数（__init__ 方法）。
        # 参数 self.block 和 self.layers 被传递给 父类ResNet 的构造函数。这些参数定义了要构建的 ResNet
        # 模型的类型（基础块是 BasicBlock 还是 Bottleneck）以及每个部分应有的块的数量。例如，如果选择
        # depth=50，self.block 将会是 Bottleneck，而 self.layers 将会是一个元组 (3, 4, 6, 3)，这指定了
        # ResNet-50 模型中每个部分的块数。
        self.init_weights(pretrained=True, cutmix=cutmix)
        self.fc = MHA(num_heads, lam, input_dim, num_classes)
        # 多头注意力分类器(MHA)  来融合特征和执行分类。
        self.loss_func = F.binary_cross_entropy_with_logits   #二元交叉熵损失

    def backbone(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # 没有avg 和 view的原因是， 后面csra要把这两个 加到一起。
        return x

    def forward_train(self, x, target):
        x = self.backbone(x)
        logit = self.fc(x)
        loss = self.loss_func(logit, target, reduction="mean")
        return logit, loss

    def forward_test(self, x):
        x = self.backbone(x)
        x = self.fc(x)
        return x

    def forward(self, x, target=None):  #根据是否提供了目标值，决定调用训练还是测试方法。
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



# 总结来说，ResNet_CSRA 类创建了一个定制的ResNet模型，其中包含了对于常规分类任务的修改，
# 并引入了基于注意力的机制来提升性能。这显示了如何在现有的深度学习模型上进行扩展和定制以满足特定的研究或应用需求。