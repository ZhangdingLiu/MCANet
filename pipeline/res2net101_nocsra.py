import torch
import torch.nn as nn
import torch.nn.functional as F
from res2net.res2net import res2net101_26w_4s
import torchvision.transforms as transforms

class Res2Net101_Multilabel(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(Res2Net101_Multilabel, self).__init__()
        # 加载预训练的Res2Net模型
        self.res2net = res2net101_26w_4s(pretrained=pretrained)
        # 替换分类器部分以适应多标签分类任务
        self.res2net.fc = nn.Linear(2048, num_classes)
        # 二元交叉熵损失函数
        # self.loss_func = nn.BCEWithLogitsLoss()  # 在外部定义损失函数时更加直观
        self.loss_func = F.binary_cross_entropy_with_logits  # 换成这个损失函数

    def extractfeature(self, x):   # 修改 需要。。。。
        # 前向传播到 layer4
        x = self.res2net.conv1(x)
        x = self.res2net.bn1(x)
        x = self.res2net.relu(x)
        x = self.res2net.maxpool(x)
        x = self.res2net.layer1(x)
        x = self.res2net.layer2(x)
        x = self.res2net.layer3(x)
        feature_map = self.res2net.layer4(x)  # layer4 的输出
        return feature_map  # 返回 feature map， 维度 (batch_size, 2048, 7, 7)


    def forward(self, x, target=None):
        x = self.res2net(x)

        if target is not None:
            # 如果提供了目标值，则计算损失
            loss = self.loss_func(x, target)
            return x, loss
        else:
            # 否则返回预测结果
            # x = torch.sigmoid(x)  # 使用sigmoid激活函数
            return x

# 无需单独调用 self.res2net.fc(x)，直接使用 self.res2net(x)
# 即可完成整个 Res2Net 模型的前向计算，包括最后的分类层。
