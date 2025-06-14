import torch
import random
import torch.nn as nn
import numpy as np

import torch
import torch.nn.functional as F

import torch
import torch.nn.functional as F

# burgers -- 计算空间导数
kernel_x = torch.tensor([[[[[-1, 0, 1]]]]], dtype=torch.float32).expand(1, 1, 1, 1, 3) / 2
kernel_y = torch.tensor([[[[[-1], [0], [1]]]]], dtype=torch.float32).expand(1, 1, 1, 3, 1) / 2
kernel_xx = torch.tensor([[[[[1, -2, 1]]]]], dtype=torch.float32).expand(1, 1, 1, 1, 3)
kernel_yy = torch.tensor([[[[[1], [-2], [1]]]]], dtype=torch.float32).expand(1, 1, 1, 3, 1)

# def compute_burgers_residual(u, v, nu):
#     # 计算空间导数
#
#     # 使用conv3d进行卷积
#     u_x = F.conv3d(u, kernel_x, padding=(0, 0, 1))
#     u_y = F.conv3d(u, kernel_y, padding=(0, 1, 0))
#     v_x = F.conv3d(v, kernel_x, padding=(0, 0, 1))
#     v_y = F.conv3d(v, kernel_y, padding=(0, 1, 0))
#
#     u_xx = F.conv3d(u, kernel_xx, padding=(0, 0, 1))
#     u_yy = F.conv3d(u, kernel_yy, padding=(0, 1, 0))
#     v_xx = F.conv3d(v, kernel_xx, padding=(0, 0, 1))
#     v_yy = F.conv3d(v, kernel_yy, padding=(0, 1, 0))
#
#     # 计算 Burgers 方程的残差
#     residual_u = u * u_x + v * u_y - nu * (u_xx + u_yy)
#     residual_v = u * v_x + v * v_y - nu * (v_xx + v_yy)
#
#     return torch.mean(residual_u.pow(2) + residual_v.pow(2))
#
#
# def burgers_loss(img_real, img_out, nu=7.704e-5):  # nu = 0.01/np.pi
#     # 计算 MSE 损失
#     # mse_loss = F.mse_loss(img_out, img_real)
#
#     # 计算基于预测与真实图像之差的 Burgers 残差损失
#     difference = img_out - img_real
#     u_diff = difference[:, 0:1, :, :, :]
#     v_diff = difference[:, 1:2, :, :, :]
#     residual_loss = compute_burgers_residual(u_diff, v_diff, nu)
#
#     # 组合总损失
#     # total_loss = mse_loss * lambada + residual_loss * lambda_burgers
#     # return total_loss
#     return residual_loss

def compute_burgers_residual(u, v, nu):
    """
    使用卷积来高效计算 u 和 v 的 Burgers 方程残差，适用于大数据量的处理
    u, v: 输入的风速分量差值
    nu: 运动粘度
    """
    # 使用 conv3d 进行一阶和二阶导数的卷积计算
    u_x = F.conv3d(u, kernel_x.to(u.device), padding=(0, 0, 1))
    u_y = F.conv3d(u, kernel_y.to(u.device), padding=(0, 1, 0))
    v_x = F.conv3d(v, kernel_x.to(u.device), padding=(0, 0, 1))
    v_y = F.conv3d(v, kernel_y.to(u.device), padding=(0, 1, 0))

    u_xx = F.conv3d(u, kernel_xx.to(u.device), padding=(0, 0, 1))
    u_yy = F.conv3d(u, kernel_yy.to(u.device), padding=(0, 1, 0))
    v_xx = F.conv3d(v, kernel_xx.to(u.device), padding=(0, 0, 1))
    v_yy = F.conv3d(v, kernel_yy.to(u.device), padding=(0, 1, 0))

    # 计算 Burgers 方程的残差
    residual_u = u * u_x + v * u_y - nu * (u_xx + u_yy)
    residual_v = u * v_x + v * v_y - nu * (v_xx + v_yy)

    # 返回 L2 范数作为残差损失
    return torch.mean(residual_u.pow(2) + residual_v.pow(2))

# def burgers_loss(img_real, img_out, nu=7.704e-5):
def burgers_loss(img_real, img_out, nu=2.3e-5):
# def burgers_loss(img_real, img_out, nu=2.16e-5):

    """
    基于卷积操作的 Burgers 方程残差计算，用于大规模数据
    img_real: 真实的风速场数据
    img_out: 预测的风速场数据
    nu: 运动粘度
    """
    # 计算真实和预测之间的差值
    difference = img_out - img_real

    # 提取差值中的 u 和 v 分量
    u_diff = difference[:, 0:1, :, :, :]
    v_diff = difference[:, 1:2, :, :, :]

    # 计算基于差值的 Burgers 残差
    residual_loss = compute_burgers_residual(u_diff, v_diff, nu)

    # 返回残差损失
    return residual_loss


class TripletLoss(nn.Module):
    '''
    Compute normal triplet loss or soft margin triplet loss given triplets
    '''
    def __init__(self, margin=None):
        super(TripletLoss, self).__init__()
        self.margin = margin
        if self.margin is None:  # if no margin assigned, use soft-margin
            self.Loss = nn.SoftMarginLoss()
        else:
            self.Loss = nn.TripletMarginLoss(margin=margin, p=2)

    def forward(self, anchor, pos, neg):
        if self.margin is None:
            num_samples = anchor.shape[0]
            y = torch.ones((num_samples, 1)).view(-1)
            if anchor.is_cuda: y = y.cuda()
            ap_dist = torch.norm(anchor-pos, 2, dim=1).view(-1)
            an_dist = torch.norm(anchor-neg, 2, dim=1).view(-1)
            loss = self.Loss(an_dist - ap_dist, y)
        else:
            loss = self.Loss(anchor, pos, neg)

        return loss

def bce_loss(input, target):
    """
    Numerically stable version of the binary cross-entropy loss function.
    As per https://github.com/pytorch/pytorch/issues/751
    See the TensorFlow docs for a derivation of this formula:
    https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits
    Input:
    - input: PyTorch Tensor of shape (N, ) giving scores.
    - target: PyTorch Tensor of shape (N,) containing 0 and 1 giving targets.

    Output:
    - A PyTorch Tensor containing the mean BCE loss over the minibatch of
      input data.
    """
    neg_abs = -input.abs()
    loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
    return loss.mean()


def gan_g_loss(scores_fake):
    """
    Input:
    - scores_fake: Tensor of shape (N,) containing scores for fake samples

    Output:
    - loss: Tensor of shape (,) giving GAN generator loss
    """
    y_fake = torch.ones_like(scores_fake) * random.uniform(0.7, 1.2)
    return bce_loss(scores_fake, y_fake)


def gan_d_loss(scores_real, scores_fake):
    """
    Input:
    - scores_real: Tensor of shape (N,) giving scores for real samples
    - scores_fake: Tensor of shape (N,) giving scores for fake samples

    Output:
    - loss: Tensor of shape (,) giving GAN discriminator loss
    """
    y_real = torch.ones_like(scores_real) * random.uniform(0.7, 1.2)
    y_fake = torch.zeros_like(scores_fake) * random.uniform(0, 0.3)
    loss_real = bce_loss(scores_real, y_real)
    loss_fake = bce_loss(scores_fake, y_fake)
    return loss_real + loss_fake


def l2_loss(pred_traj, pred_traj_gt, loss_mask, random=0, mode='average'):
    """
    Input:
    - pred_traj: Tensor of shape (seq_len, batch, 2). Predicted trajectory.
    - pred_traj_gt: Tensor of shape (seq_len, batch, 2). Groud truth
    predictions.
    - loss_mask: Tensor of shape (batch, seq_len)
    - mode: Can be one of sum, average, raw
    Output:
    - loss: l2 loss depending on mode
    """
    seq_len, batch, _ = pred_traj.size()
    loss = (loss_mask.unsqueeze(dim=2) *
            (pred_traj_gt.permute(1, 0, 2) - pred_traj.permute(1, 0, 2))**2)
    if mode == 'sum':
        return torch.sum(loss)
    elif mode == 'average':
        return torch.sum(loss) / torch.numel(loss_mask.data)
    elif mode == 'raw':
        return loss.sum(dim=2).sum(dim=1)

def toNE(pred_traj,pred_Me):
    # 0  经度  1纬度
    pred_traj[:, :,0] = pred_traj[:, :,0] / 10 * 500 + 1300
    pred_traj[:,:,1] = pred_traj[:,:,1] / 6 * 300 + 300
    # 0 气压 1 风速
    pred_Me[:, :, 0] = pred_Me[:, :, 0]* 50 + 960
    pred_Me[:, :, 1] = pred_Me[:, :, 1] * 25 + 40
    return pred_traj,pred_Me

def trajectory_displacement_error(pred_traj, pred_traj_gt, consider_ped=None, mode='sum'):
    """
    Input:
    - pred_traj: Tensor of shape (seq_len, batch, 2). Predicted trajectory.
    - pred_traj_gt: Tensor of shape (seq_len, batch, 2). Ground truth
    predictions.
    - consider_ped: Tensor of shape (batch)
    - mode: Can be one of sum, raw
    Output:
    - loss: gives the eculidian displacement error
    """
    seq_len, _, _ = pred_traj.size()
    loss = pred_traj_gt.permute(1, 0, 2) - pred_traj.permute(1, 0, 2)
    loss[:,:,0] = (loss[:,:,0]/10)*111*(pred_traj_gt.permute(1, 0, 2)[:,:,1]/10*np.pi/180).cos()
    # w0 = pred_traj_gt.permute(1, 0, 2)[:,:,1]/10
    # w = (pred_traj_gt.permute(1, 0, 2)[:,:,1]/10*np.pi/180)
    # w2 = (pred_traj_gt.permute(1, 0, 2)[:, :, 1] / 10 * np.pi / 180).cos()
    loss[:, :, 1] = (loss[:, :, 1] / 10) * 111
    loss = loss**2
    loss = torch.sqrt(loss[:,:,0]+loss[:,:,1])
    # if consider_ped is not None:
    #     loss = torch.sqrt(loss.sum(dim=2)).sum(dim=1) * consider_ped
    # else:
    #     loss = torch.sqrt(loss.sum(dim=2)).sum(dim=1)
    if mode == 'sum':
        return torch.sum(loss)
    elif mode == 'raw':
        return loss

def trajectory_diff(pred_traj, pred_traj_gt, consider_ped=None, mode='sum'):
    """
    Input:
    - pred_traj: Tensor of shape (seq_len, batch, 2). Predicted trajectory.
    - pred_traj_gt: Tensor of shape (seq_len, batch, 2). Ground truth
    predictions.
    - consider_ped: Tensor of shape (batch)
    - mode: Can be one of sum, raw
    Output:
    - loss: gives the eculidian displacement error
    """
    seq_len, _, _ = pred_traj.size()
    loss = pred_traj.permute(1, 0, 2) - pred_traj_gt.permute(1, 0, 2)
    loss[:,:,0] = (loss[:,:,0]/10)*111
    # w0 = pred_traj_gt.permute(1, 0, 2)[:,:,1]/10
    # w = (pred_traj_gt.permute(1, 0, 2)[:,:,1]/10*np.pi/180)
    # w2 = (pred_traj_gt.permute(1, 0, 2)[:, :, 1] / 10 * np.pi / 180).cos()
    loss[:, :, 1] = (loss[:, :, 1] / 10) * 111*(pred_traj_gt.permute(1, 0, 2)[:,:,1]/10*np.pi/180).cos()
    # loss = loss**2
    # loss = torch.sqrt(loss[:,:,0]+loss[:,:,1])
    # if consider_ped is not None:
    #     loss = torch.sqrt(loss.sum(dim=2)).sum(dim=1) * consider_ped
    # else:
    #     loss = torch.sqrt(loss.sum(dim=2)).sum(dim=1)
    if mode == 'sum':
        return torch.sum(loss)
    elif mode == 'raw':
        return loss

def value_diff(pred_traj, pred_traj_gt, consider_ped=None, mode='sum'):
    """
    Input:
    - pred_traj: Tensor of shape (seq_len, batch, 2). Predicted trajectory.
    - pred_traj_gt: Tensor of shape (seq_len, batch, 2). Ground truth
    predictions.
    - consider_ped: Tensor of shape (batch)
    - mode: Can be one of sum, raw
    Output:
    - loss: gives the eculidian displacement error
    """
    seq_len, _, _ = pred_traj.size()
    loss = pred_traj.permute(1, 0, 2)-pred_traj_gt.permute(1, 0, 2)

    # loss = loss**2
    # loss = torch.sqrt(loss[:,:,0]+loss[:,:,1])
    # if consider_ped is not None:
    #     loss = torch.sqrt(loss.sum(dim=2)).sum(dim=1) * consider_ped
    # else:
    #     loss = torch.sqrt(loss.sum(dim=2)).sum(dim=1)
    if mode == 'sum':
        return torch.sum(loss)
    elif mode == 'raw':
        return loss

def value_error(pred_traj, pred_traj_gt, consider_ped=None, mode='sum'):
    """
    Input:
    - pred_traj: Tensor of shape (seq_len, batch, 2). Predicted trajectory.
    - pred_traj_gt: Tensor of shape (seq_len, batch, 2). Ground truth
    predictions.
    - consider_ped: Tensor of shape (batch)
    - mode: Can be one of sum, raw
    Output:
    - loss: gives the eculidian displacement error
    """
    seq_len, _, _ = pred_traj.size()
    loss = torch.abs((pred_traj.permute(1, 0, 2)-pred_traj_gt.permute(1, 0, 2)))

    # loss = loss**2
    # loss = torch.sqrt(loss[:,:,0]+loss[:,:,1])
    # if consider_ped is not None:
    #     loss = torch.sqrt(loss.sum(dim=2)).sum(dim=1) * consider_ped
    # else:
    #     loss = torch.sqrt(loss.sum(dim=2)).sum(dim=1)
    if mode == 'sum':
        return torch.sum(loss)
    elif mode == 'raw':
        return loss


def displacement_error(pred_traj, pred_traj_gt, consider_ped=None, mode='sum'):
    """
    Input:
    - pred_traj: Tensor of shape (seq_len, batch, 2). Predicted trajectory.
    - pred_traj_gt: Tensor of shape (seq_len, batch, 2). Ground truth
    predictions.
    - consider_ped: Tensor of shape (batch)
    - mode: Can be one of sum, raw
    Output:
    - loss: gives the eculidian displacement error
    """
    seq_len, _, _ = pred_traj.size()
    loss = pred_traj_gt.permute(1, 0, 2) - pred_traj.permute(1, 0, 2)
    loss = loss**2
    if consider_ped is not None:
        loss = torch.sqrt(loss.sum(dim=2)).sum(dim=1) * consider_ped
    else:
        loss = torch.sqrt(loss.sum(dim=2)).sum(dim=1)
    if mode == 'sum':
        return torch.sum(loss)
    elif mode == 'raw':
        return loss


def final_displacement_error(
    pred_pos, pred_pos_gt, consider_ped=None, mode='sum'
):
    """
    Input:
    - pred_pos: Tensor of shape (batch, 2). Predicted last pos.
    - pred_pos_gt: Tensor of shape (seq_len, batch, 2). Groud truth
    last pos
    - consider_ped: Tensor of shape (batch)
    Output:
    - loss: gives the eculidian displacement error
    """
    loss = pred_pos_gt - pred_pos
    loss = loss**2
    if consider_ped is not None:
        loss = torch.sqrt(loss.sum(dim=1)) * consider_ped
    else:
        loss = torch.sqrt(loss.sum(dim=1))
    if mode == 'raw':
        return loss
    else:
        return torch.sum(loss)
