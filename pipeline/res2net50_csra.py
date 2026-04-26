from .csra import CSRA, MHA

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from res2net.res2net import Res2Net, Bottle2neck, model_urls

import torch.nn.functional as F


class CustomRes2Net(Res2Net):
    def __init__(self, block, layers, baseWidth=26, scale=4, pretrained=False, **kwargs):
        super(CustomRes2Net, self).__init__(block, layers, baseWidth=baseWidth, scale=scale, **kwargs)

        # 手动加载预训练模型的权重
        if pretrained:
            model_url = model_urls['res2net50_26w_4s']  # 确保你选择了正确的 URL
            state_dict = model_zoo.load_url(model_url)
            self.load_state_dict(state_dict)

    def forward(self, x):
        # 使用标准的 Res2Net 前向传播，但移除 avgpool 和 view 操作
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # 不执行 avgpool 和 view，直接返回特征图
        return x


class Res2Net50_Csra(nn.Module):
    def __init__(self, num_heads, lam, num_classes, input_dim=2048, pretrained=True):
        super(Res2Net50_Csra, self).__init__()
        # 使用定制的 Res2Net 模型
        self.res2net = CustomRes2Net(Bottle2neck, [3, 4, 6, 3], baseWidth=26, scale=4, pretrained=pretrained)

        # 替换分类器部分为 MHA
        self.res2net.fc = MHA(num_heads, lam, input_dim, num_classes)
        #需要细看这里输出的是什么

        # 二元交叉熵损失函数
        self.loss_func = F.binary_cross_entropy_with_logits

    def forward(self, x, target=None):
        x = self.res2net(x)  # 只会执行卷积层和残差层的前向传播（即 conv1 到 layer4），然后直接输出特征图。

        # 通过 MHA 进行分类，得到 (B, num_classes)
        x = self.res2net.fc(x)   #这个地方要单独加上

        if target is not None:
            # 如果提供了目标值，则计算损失
            loss = self.loss_func(x, target, reduction="mean")
            return x, loss
        else:
            # 否则返回预测结果
            return x
