import torch
import torch.nn.functional as F
from math import pi
import numpy as np
import torch.autograd

# euler batch*4
# output cuda batch*3*3 matrices in the rotation order of XZ'Y'' (intrinsic) or YZX (extrinsic)
def compute_rotation_matrix_from_euler(euler):
    batch = euler.shape[0]

    c1 = torch.cos(euler[:, 0]).view(batch, 1)  # batch*1
    s1 = torch.sin(euler[:, 0]).view(batch, 1)  # batch*1
    c2 = torch.cos(euler[:, 2]).view(batch, 1)  # batch*1
    s2 = torch.sin(euler[:, 2]).view(batch, 1)  # batch*1
    c3 = torch.cos(euler[:, 1]).view(batch, 1)  # batch*1
    s3 = torch.sin(euler[:, 1]).view(batch, 1)  # batch*1

    row1 = torch.cat((c2 * c3, -s2, c2 * s3), 1).view(-1, 1, 3)  # batch*1*3
    row2 = torch.cat((c1 * s2 * c3 + s1 * s3, c1 * c2, c1 * s2 * s3 - s1 * c3), 1).view(-1, 1, 3)  # batch*1*3
    row3 = torch.cat((s1 * s2 * c3 - c1 * s3, s1 * c2, s1 * s2 * s3 + c1 * c3), 1).view(-1, 1, 3)  # batch*1*3

    matrix = torch.cat((row1, row2, row3), 1)  # batch*3*3

    return matrix


def compute_correlation_volume_weight(fmap1, fmap2):
    batch, dim, ht, wd = fmap1.shape
    fmap1 = fmap1.view(batch, dim, ht * wd)
    fmap2 = fmap2.view(batch, dim, ht * wd)
    fmap1 = F.normalize(fmap1,dim=1)
    fmap2 = F.normalize(fmap2,dim=1)

    corr = torch.matmul(fmap1.transpose(1, 2), fmap2)
    # corr = corr.view(batch, ht, wd, 1, ht, wd)
    # corr = corr / torch.sqrt(torch.tensor(dim).float())

    return corr

def compute_correlation_volume_x0_y0_x1_y1(wd,ht,K0,K1,batch_size):
    x0 = np.arange(wd)
    x0 = (x0 + 0.5) * 8 - 0.5
    y0 = np.arange(ht)
    y0 = (y0 + 0.5) * 8 - 0.5

    # 笛卡尔积
    from itertools import product
    volume_list = list(product(x0, y0))
    corr_volume_x0_y0 = np.array(volume_list)
    corr_volume_x0_y0 = np.insert(corr_volume_x0_y0, 2, values=1, axis=1).reshape(ht * wd, 3, 1)

    depth_K0_np = K0.detach().cpu().numpy()
    depth_K1_np = K1.detach().cpu().numpy()
    depth_K0_np_inv = np.linalg.inv(depth_K0_np)
    depth_K1_np_inv = np.linalg.inv(depth_K1_np)
    corr_volume_x0_y0_cam = depth_K0_np_inv[:, np.newaxis, :, :] @ corr_volume_x0_y0[None]
    corr_volume_x1_y1_cam = depth_K1_np_inv[:, np.newaxis, :, :] @ corr_volume_x0_y0[None]
    corr_volume_x0_y0_cam = corr_volume_x0_y0_cam.reshape(batch_size, wd * ht, 3)[:, :, :2]
    corr_volume_x1_y1_cam = corr_volume_x1_y1_cam.reshape(batch_size, wd * ht, 3)[:, :, :2]

    corr_volume_x0_y0_x1_y1 = list(product(corr_volume_x0_y0_cam[0], corr_volume_x1_y1_cam[0]))
    corr_volume_x0_y0_x1_y1 = np.array(corr_volume_x0_y0_x1_y1).reshape(-1, 4)[None]  # 1, w*h*w*h, 4
    return corr_volume_x0_y0_x1_y1


