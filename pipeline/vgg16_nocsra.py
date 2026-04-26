import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16
import torch.utils.model_zoo as model_zoo



class VGG16_Multilabel(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(VGG16_Multilabel, self).__init__()
        # 加载预训练的VGG16模型
        self.vgg = vgg16(pretrained=pretrained)
        # 替换分类器部分以适应多标签分类任务
        self.vgg.classifier[-1] = nn.Linear(4096, num_classes)
        # 二元交叉熵损失函数
        self.loss_func = F.binary_cross_entropy_with_logits

    def forward(self, x, target=None):
        x = self.vgg.features(x)
        x = self.vgg.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.vgg.classifier(x)

        if target is not None:
            # 如果提供了目标值，则计算损失
            loss = self.loss_func(x, target)
            return x, loss
        else:
            # 否则返回预测结果
            x = torch.sigmoid(x)  # 使用sigmoid激活函数
            return x

    # def init_weights(self, pretrained=True):
    #     if pretrained:
    #         print("Initializing weights from pre-trained model")
    #         model_url = model_urls['vgg16']
    #         state_dict = model_zoo.load_url(model_url)
    #         self.vgg.load_state_dict(state_dict)

# 1. init_weights 方法的多余
# 在 VGG16_Multilabel 类的 __init__ 方法中，您已经通过 pretrained=True 参数加载了预训练的模型权重。因此，init_weights 方法实际上是多余的，除非您希望在某些情况下手动加载权重。
# 如果您希望保留 init_weights 方法，确保不会重复加载预训练权重。

