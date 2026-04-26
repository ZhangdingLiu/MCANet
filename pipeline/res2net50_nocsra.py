import torch
import torch.nn as nn
# import torch.nn.functional as F
from res2net.res2net import res2net50
import torchvision.transforms as transforms


class Res2Net50_Multilabel(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(Res2Net50_Multilabel, self).__init__()
        # 加载预训练的Res2Net模型
        self.res2net = res2net50(pretrained=pretrained)
        # 替换分类器部分以适应多标签分类任务
        self.res2net.fc = nn.Linear(2048, num_classes)     # 这里是2048吗？
        # 二元交叉熵损失函数
        self.loss_func = nn.BCEWithLogitsLoss()  # 在外部定义损失函数时更加直观

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
