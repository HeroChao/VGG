

import time
import torch
from torch import nn, optim
import torchvision

# 一个 vgg 的 block，传入三个参数，第一个是模型层数，第二个是输入的通道数，第三个是输出的通道数，第一
# 层卷积接受的输入通道就是图片输入的通道数，然后输出最后的输出通道数，后面的卷积接受的通道数就是最后
# 的输出通道数
def vgg_block(num_convs, in_channels, out_channels):
    net = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), nn.ReLU(True)]  # 定义第一层

    for i in range(num_convs - 1):  # 定义后面的很多层
        net.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
        net.append(nn.ReLU(True))

    net.append(nn.MaxPool2d(2, 2))  # 定义池化层
    return nn.Sequential(*net)


# 函数对 vgg block 进行堆叠
def vgg_stack(num_convs, channels):
    net = []
    for n, c in zip(num_convs, channels):
        in_c = c[0]
        out_c = c[1]
        net.append(vgg_block(n, in_c, out_c))
    return nn.Sequential(*net)


# vgg模型
class vgg(nn.Module):
    def __init__(self):
        super(vgg, self).__init__()
        self.feature = vgg_net
        self.fc = nn.Sequential(
            nn.Linear(512, 100),
            nn.ReLU(True),
            nn.Linear(100, 10)
        )

    def forward(self, x):
        x = self.feature(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x