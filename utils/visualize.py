from PIL import Image
import json
import torch
from torchvision import transforms
import cv2
import numpy as np
import os
import torch.nn as nn

# 这段代码实现了从图像中读取数据，
# 并将类激活映射（Class Activation Mapping, CAM）叠加在原始图像上，用以可视化模型预测的焦点区域。

def show_cam_on_img(img, mask, img_path_save):
    heat_map = cv2.applyColorMap(np.uint8(255*mask), cv2.COLORMAP_JET)
    heat_map = np.float32(heat_map) / 255

    cam = heat_map + np.float32(img)
    cam = cam / np.max(cam)
    cv2.imwrite(img_path_save, np.uint8(255 * cam))

img_path_read = ""
img_path_save = ""

def main():
  img = cv2.imread(img_path_read, flags=1)
  img = np.float32(cv2.resize(img, (224, 224))) / 255   # 448可以吗？ original是224

  # cam_all is the score tensor of shape (B, C, H, W), similar to y_raw in out Figure 1
  # cls_idx specifying the i-th class out of C class
  # visualize the 0's class heatmap
  cls_idx = 0
  cam = cam_all[cls_idx]

  # cam = nn.ReLU()(cam)
  cam = cam / torch.max(cam)
  
  cam = cv2.resize(np.array(cam), (224, 224))
  show_cam_on_img(img, cam, img_path_save)

# 存在的问题
# 变量 cam_all 没有在代码中定义，它应该是从某个模型的输出获取的。
# 图像路径 img_path_read 和 img_path_save 都是空字符串，实际使用时需要指定为有效的文件路径。
# 使用 torch.max(cam) 进行归一化时，应确保 cam 中有非零元素以避免除以零的错误。

# 这段代码的目标是展示如何通过类激活映射来可视化模型对图像
# 中某一类别的响应区域，这对于理解和解释卷积神经网络的决策过程非常有用。