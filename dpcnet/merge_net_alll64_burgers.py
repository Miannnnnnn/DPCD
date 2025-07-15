import torch
from torch import nn
import torch.nn.functional as F

class Merge_Net_All(nn.Module):
    def __init__(self):
        super(Merge_Net_All, self).__init__()
        self.att1 = nn.Sequential(
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=1, padding=0),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=1, padding=0),
            nn.BatchNorm3d(64),
            nn.Sigmoid(),
        )
        self.att2 = nn.Sequential(
            nn.Conv3d(in_channels=128, out_channels=128, kernel_size=1, padding=0),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=128, out_channels=128, kernel_size=1, padding=0),
            nn.BatchNorm3d(128),
            nn.Sigmoid(),
        )
        self.att3 = nn.Sequential(
            nn.Conv3d(in_channels=256, out_channels=256, kernel_size=1, padding=0),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=256, out_channels=256, kernel_size=1, padding=0),
            nn.BatchNorm3d(256),
            nn.Sigmoid(),
        )

        self.cross_unit = nn.Parameter(data=torch.ones(4, 6))

        self.fuse_unit = nn.Parameter(data=torch.ones(4, 4))

        self.conv1 = nn.Conv3d(2, 64, kernel_size=(1, 3, 3), padding=(1, 1, 1))
        self.pool = nn.MaxPool3d(2, 2)
        self.conv2 = nn.Conv3d(64, 128, kernel_size=(1, 3, 3), padding=(1, 1, 1))
        self.conv3 = nn.Conv3d(128, 256, kernel_size=(1, 3, 3), padding=(1, 1, 1))
        self.conv4 = nn.Conv3d(256, 512, kernel_size=(1, 3, 3), padding=(1, 1, 1))
        #self.conv1x1 = nn.Conv3d(48, 12, kernel_size=1)
        self.conv1x1 = nn.Conv3d(48, 12, kernel_size=1, stride=(4, 1, 1))

        self.fc1 = nn.Linear(8, 1)
        self.fc2 = nn.Linear(256 * 4, 256)
        self.fc = nn.Linear(32,12)

        # 第一步：转置卷积，从 8x8 到 16x16，减少通道数从 256 到 128
        self.up1 = nn.ConvTranspose3d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        # 第二步：转置卷积，从 16x16 到 32x32，减少通道数从 128 到 64
        self.up2 = nn.ConvTranspose3d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        # 第三步：转置卷积，从 32x32 到 64x64，调整通道数从 64 到 12
        self.up3 = nn.ConvTranspose3d(64, 12, kernel_size=3, stride=2, padding=1, output_padding=1)

        # 添加用于残差连接的1x1卷积层
        self.skip1 = nn.Conv3d(256, 128, kernel_size=1)
        self.skip2 = nn.Conv3d(128, 64, kernel_size=1)
        self.skip3 = nn.Conv3d(64, 12, kernel_size=1)


    def forward(self, img_u, img_v):
        seq_list = []
        for i in range(4):
            u = img_u[:, :, i * 2:(i + 1) * 2, :, :].permute(0, 2, 1, 3, 4)  # 64, 2, 64, 64
            v = img_v[:, :, i * 2:(i + 1) * 2, :, :].permute(0, 2, 1, 3, 4)

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

            time_seq1 = F.relu(self.up1(time_seq))
            time_seq2 = F.relu(self.up2(time_seq1))
            time_seq = self.up3(time_seq2)

            seq_list.append(time_seq)

        uv = torch.cat(seq_list, dim=1)
       
        uv = self.conv1x1(uv)
        uv = uv.permute(0, 2, 1, 3, 4)

        return uv

if __name__ == '__main__':
    img_u = torch.randn(64,1,8,64,64).cuda()
    img_v = torch.randn(64,1,8,64,64).cuda()
    net = Merge_Net_All().cuda()
    out = net(img_u, img_v)
    print(out.shape)