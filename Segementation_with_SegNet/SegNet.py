import torch
import torch.nn as nn
from torchinfo import summary

BATCH_MOMENTUM = 0.9
PADDING = 1


def double_conv(in_channels, out_channels, kernel_size, up):
    if up == 0:
        conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=PADDING),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=PADDING),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    else:
        conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size, padding=PADDING),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=PADDING),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    return conv


def triple_conv(in_channels, out_channels, kernel_size, up):
    if up == 0:
        conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=PADDING),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=PADDING),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=PADDING),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    else:
        conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size, padding=PADDING),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size, padding=PADDING),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=PADDING),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    return conv


class SegNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SegNet, self).__init__()

        self.max_pool_2x2 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.max_unpool_2x2 = nn.MaxUnpool2d(kernel_size=2, stride=2)

        # down convolution
        self.donw_conv1 = double_conv(in_channels, 64, 3, 0)  # arguments, in_channels, out_channels, kernel size=3, down conv if it is 0
        self.down_conv2 = double_conv(64, 128, 3, 0)
        self.down_conv3 = triple_conv(128, 256, 3, 0)
        self.down_conv4 = triple_conv(256, 512, 3, 0)
        self.down_conv5 = triple_conv(512, 512, 3, 0)

        # up convolution
        self.up_conv1 = triple_conv(512, 512, 3, 1)
        self.up_conv2 = triple_conv(512, 256, 3, 1)
        self.up_conv3 = triple_conv(256, 128, 3, 1)
        self.up_conv4 = double_conv(128, 64, 3, 1)
        self.up_conv5 = nn.Conv2d(64, out_channels, 3, padding=PADDING)

    def forward(self, x):

        # encoder
        x1 = self.donw_conv1(x)
        x2, ind1 = self.max_pool_2x2(x1)
        size1 = x2.size()
        # print(size1)

        x3 = self.down_conv2(x2)
        x4, ind2 = self.max_pool_2x2(x3)
        size2 = x4.size()
        # print(size2)

        x5 = self.down_conv3(x4)
        x6, ind3 = self.max_pool_2x2(x5)
        size3 = x6.size()
        # print(size3)

        x7 = self.down_conv4(x6)
        x8, ind4 = self.max_pool_2x2(x7)
        size4 = x8.size()
        # print(size4)

        x9 = self.down_conv5(x8)
        x10, ind5 = self.max_pool_2x2(x9)
        # size5 = x10.size()
        # print(size5)

        # decoder

        x11 = self.max_unpool_2x2(x10, ind5, output_size=size4)
        x12 = self.up_conv1(x11)
        # print(x12.size())

        x13 = self.max_unpool_2x2(x12, ind4, output_size=size3)
        x14 = self.up_conv2(x13)
        # print(x14.size())

        x15 = self.max_unpool_2x2(x14, ind3, output_size=size2)
        x16 = self.up_conv3(x15)
        # print(x16.size())

        x17 = self.max_unpool_2x2(x16, ind2, output_size=size1)
        x18 = self.up_conv4(x17)
        # print(x18.size())

        x19 = self.max_unpool_2x2(x18, ind1)
        x20 = self.up_conv5(x19)
        # print(x20.size())
        return x20


def test():
    x = torch.randn((1, 3, 160, 160))
    model = SegNet(in_channels=3, out_channels=1)
    preds = model(x)
    print(preds.shape)
    print(x.shape)
    assert preds.shape == x.shape


if __name__ == "__main__":
    test()








