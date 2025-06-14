import torch
from torch import nn
import torch.nn.functional as F

class uvNet(nn.Module):
    def __init__(self, in_channels):
        super(uvNet, self).__init__()
        self.dcn = nn.DeformableConv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.q_linear = nn.Linear(in_channels, in_channels)
        self.k_linear = nn.Linear(in_channels, in_channels)
        self.v_linear = nn.Linear(in_channels, in_channels)
        self.scale = in_channels ** -0.5

    def forward(self, u, v):
        # 可变形卷积处理
        u = self.dcn(u)
        v = self.dcn(v)

        # 计算注意力
        b, c, h, w = u.shape
        q_u = self.q_linear(u.permute(0, 2, 3, 1).view(b, -1, c)).transpose(1, 2)
        k_v = self.k_linear(v.permute(0, 2, 3, 1).view(b, -1, c))
        v_v = self.v_linear(v.permute(0, 2, 3, 1).view(b, -1, c)).transpose(1, 2)

        attn = torch.bmm(q_u, k_v) * self.scale
        attn = F.softmax(attn, dim=-1)

        out = torch.bmm(attn, v_v).view(b, c, h, w)

        return out



