import torch
import torch.nn as nn
import torch.nn.functional as F

class MyAlex(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            # 第一层卷积
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),  # 输入: 1x12x12, 输出: 64x12x12
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 输出: 64x6x6

            # 第二层卷积
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # 输入: 64x6x6, 输出: 128x6x6
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 输出: 128x3x3

            # 第三层卷积
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),  # 输入: 128x3x3, 输出: 256x3x3
            nn.ReLU(inplace=True),

            # 第四层卷积
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),  # 输入: 256x3x3, 输出: 256x3x3
            nn.ReLU(inplace=True),

            # 第五层卷积
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),  # 输入: 256x3x3, 输出: 128x3x3
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 输出: 128x1x1
        )

        self.classifier = nn.Sequential(
            nn.Linear(128 * 1 * 1, 1024),  # 输入: 128, 输出: 1024
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),  # 添加 dropout
            nn.Linear(1024, 256),  # 输入: 1024, 输出: 256
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes)  # 输入: 256, 输出: num_classes
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)  # 展平特征图
        x = self.classifier(x)
        return F.softmax(x, dim=1)  # 使用 softmax 输出概率分布
    
    
# 损失函数


# 数据记录


if __name__ == "__main__":
    from torch.utils.tensorboard import SummaryWriter
    demo = MyAlex(5)
    input = torch.randn((4,1,12,12))
    writer=SummaryWriter("Logs")
    writer.add_graph(demo, input)
    writer.close()