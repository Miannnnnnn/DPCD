import torch
from torch import nn
import torch.nn.functional as F
from torchvision.ops import DeformConv2d

class DeformableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0):
        super(DeformableConv2d, self).__init__()

        self.offset_conv = nn.Conv2d(in_channels, 2 * kernel_size * kernel_size, kernel_size, stride, padding)
        self.deform_conv = DeformConv2d(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        offset = self.offset_conv(x)
        return self.deform_conv(x, offset)


class Merge_Net_All(nn.Module):
    def __init__(self):
        super(Merge_Net_All, self).__init__()

        # att1
        self.att1_conv1 = DeformableConv2d(in_channels=64, out_channels=64, kernel_size=1, padding=0)
        self.att1_bn1 = nn.BatchNorm2d(64)
        self.att1_conv2 = DeformableConv2d(in_channels=64, out_channels=64, kernel_size=1, padding=0)
        self.att1_bn2 = nn.BatchNorm2d(64)

        # att2
        self.att2_conv1 = DeformableConv2d(in_channels=128, out_channels=128, kernel_size=1, padding=0)
        self.att2_bn1 = nn.BatchNorm2d(128)
        self.att2_conv2 = DeformableConv2d(in_channels=128, out_channels=128, kernel_size=1, padding=0)
        self.att2_bn2 = nn.BatchNorm2d(128)

        # att3
        self.att3_conv1 = DeformableConv2d(in_channels=256, out_channels=256, kernel_size=1, padding=0)
        self.att3_bn1 = nn.BatchNorm2d(256)
        self.att3_conv2 = DeformableConv2d(in_channels=256, out_channels=256, kernel_size=1, padding=0)
        self.att3_bn2 = nn.BatchNorm2d(256)

        self.cross_unit = nn.Parameter(data=torch.ones(4, 6))

        self.fuse_unit = nn.Parameter(data=torch.ones(4, 4))

        self.conv1 = nn.Conv2d(2, 64, kernel_size=(3, 3), padding=(1, 1))
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=(1, 1))
        self.conv3 = nn.Conv2d(128, 256, kernel_size=(3, 3), padding=(1, 1))
        self.conv4 = nn.Conv2d(256, 512, kernel_size=(3, 3), padding=(1, 1))
        self.conv1x1 = nn.Conv2d(48, 12, kernel_size=1)

        # 第一步：转置卷积，从 8x8 到 16x16，减少通道数从 256 到 128
        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        # 第二步：转置卷积，从 16x16 到 32x32，减少通道数从 128 到 64
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        # 第三步：转置卷积，从 32x32 到 64x64，减少通道数从 64 到 12
        self.up3 = nn.ConvTranspose2d(64, 12, kernel_size=3, stride=2, padding=1, output_padding=1)

        # 第一步：转置卷积，从 8x8 到 16x16，减少通道数从 256 到 128
        # self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        # # 第二步：转置卷积，从 16x16 到 32x32，减少通道数从 128 到 64
        # self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        # # 第三步：转置卷积，从 32x32 到 64x64，调整通道数从 64 到 12
        # self.up3 = nn.ConvTranspose2d(64, 12, kernel_size=4, stride=2, padding=1)

    def att1(self, x):
        x = F.relu(self.att1_bn1(self.att1_conv1(x)), inplace=True)
        x = self.att1_bn2(self.att1_conv2(x))
        return torch.sigmoid(x)

    def att2(self, x):
        x = F.relu(self.att2_bn1(self.att2_conv1(x)), inplace=True)
        x = self.att2_bn2(self.att2_conv2(x))
        return torch.sigmoid(x)

    def att3(self, x):
        x = F.relu(self.att3_bn1(self.att3_conv1(x)), inplace=True)
        x = self.att3_bn2(self.att3_conv2(x))
        return torch.sigmoid(x)

    def forward(self, img_u, img_v):
        seq_list = []

        img_u = img_u.squeeze(1)
        img_v = img_v.squeeze(1)

        for i in range(4):
            u = img_u[:, i * 2:(i + 1) * 2, :, :]  # 64, 2, 64, 64
            v = img_v[:, i * 2:(i + 1) * 2, :, :]

            u = self.pool(F.relu(self.conv1(u)))
            u = self.att1(u) * u
            v = self.pool(F.relu(self.conv1(v)))
            v = self.att1(v) * v


            # fuse1
            time_seq_1 = self.cross_unit[i][0]/(self.cross_unit[i][0]+self.cross_unit[i][1]) * u +\
                         self.cross_unit[i][1]/(self.cross_unit[i][0]+self.cross_unit[i][1]) * v

            time_seq_1 = self.pool(F.relu(self.conv2(time_seq_1)))
            time_seq_1 = self.att2(time_seq_1) * time_seq_1

            u = self.pool(F.relu(self.conv2(u)))
            u = self.att2(u) * u
            v = self.pool(F.relu(self.conv2(v)))
            v = self.att2(v) * v

            # fuse2
            time_seq_2 = self.cross_unit[i][2]/(self.cross_unit[i][2]+self.cross_unit[i][3]) * u +\
                         self.cross_unit[i][3]/(self.cross_unit[i][2]+self.cross_unit[i][3]) * v
            time_seq_2 = self.cross_unit[i][0]/(self.cross_unit[i][0]+self.cross_unit[i][1]) * time_seq_1 +\
                         self.cross_unit[i][1]/(self.cross_unit[i][0]+self.cross_unit[i][1]) * time_seq_2
            time_seq_2 = self.pool(F.relu(self.conv3(time_seq_2)))
            time_seq_2 = self.att3(time_seq_2) * time_seq_2

            u = self.pool(F.relu(self.conv3(u)))
            u = self.att3(u) * u
            v = self.pool(F.relu(self.conv3(v)))
            v = self.att3(v) * v

            # fuse3
            time_seq = self.cross_unit[i][4]/(self.cross_unit[i][4]+self.cross_unit[i][5]) * u +\
                         self.cross_unit[i][5]/(self.cross_unit[i][4]+self.cross_unit[i][5]) * v
            time_seq = self.cross_unit[i][2]/(self.cross_unit[i][2]+self.cross_unit[i][3]) * time_seq_2 +\
                         self.cross_unit[i][3]/(self.cross_unit[i][2]+self.cross_unit[i][3]) * time_seq

            time_seq = self.att3(time_seq) * time_seq

            time_seq = F.relu(self.up1(time_seq))
            time_seq = F.relu(self.up2(time_seq))
            time_seq = self.up3(time_seq)
            seq_list.append(time_seq)

        uv = torch.cat(seq_list, dim=1)
        uv = self.conv1x1(uv)

        return uv

if __name__ == '__main__':
    img_u = torch.randn(64,1,8,64,64).cuda()
    img_v = torch.randn(64,1,8,64,64).cuda()
    net = Merge_Net_All().cuda()
    out = net(img_u, img_v)
    print(out.shape)