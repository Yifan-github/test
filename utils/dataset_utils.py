import numpy as np
import time
import random
import torch
import cv2

def dummy_collate_fn(data_list):
    return data_list[0]

def simple_collate_fn(data_list):
    ks=data_list[0].keys()
    outputs={k:[] for k in ks}
    for k in ks:
        for data in data_list:
            outputs[k].append(data[k])
        outputs[k]=torch.stack(outputs[k],0)
    return outputs

def set_seed(index,is_train):
    if is_train:
        np.random.seed((index+int(time.time())) % (2**16))
        random.seed((index+int(time.time())) % (2**16)+1)
        torch.random.manual_seed((index+int(time.time())) % (2**16)+1)
    else:
        np.random.seed(index % (2 ** 16))
        random.seed(index % (2 ** 16) + 1)
        torch.random.manual_seed(index % (2 ** 16) + 1)

def _py_additive_shade(img, nb_ellipses=20, transparency_range=[-0.5, 0.8],
                   kernel_size_range=[250, 350]):
    min_dim = min(img.shape[:2]) / 4
    mask = np.zeros(img.shape[:2], np.uint8)
    for i in range(nb_ellipses):
        ax = int(max(np.random.rand() * min_dim, min_dim / 5))
        ay = int(max(np.random.rand() * min_dim, min_dim / 5))
        max_rad = max(ax, ay)
        x = np.random.randint(max_rad, img.shape[1] - max_rad)  # center
        y = np.random.randint(max_rad, img.shape[0] - max_rad)
        angle = np.random.rand() * 90
        cv2.ellipse(mask, (x, y), (ax, ay), angle, 0, 360, 255, -1)

    transparency = np.random.uniform(*transparency_range)
    kernel_size = np.random.randint(*kernel_size_range)
    if (kernel_size % 2) == 0:  # kernel_size has to be odd
        kernel_size += 1
    mask = cv2.GaussianBlur(mask.astype(np.float32), (kernel_size, kernel_size), 0)
    shaded = img * (1 - transparency * mask[..., np.newaxis] / 255.)
    return np.clip(shaded, 0, 255)

def motion_blur(img, max_kernel_size=10):
    # Either vertial, hozirontal or diagonal blur
    mode = np.random.choice(['h', 'v', 'diag_down', 'diag_up'])
    ksize = np.random.randint(0, (max_kernel_size + 1) / 2) * 2 + 1  # make sure is odd
    center = int((ksize - 1) / 2)
    kernel = np.zeros((ksize, ksize))
    if mode == 'h':
        kernel[center, :] = 1.
    elif mode == 'v':
        kernel[:, center] = 1.
    elif mode == 'diag_down':
        kernel = np.eye(ksize)
    elif mode == 'diag_up':
        kernel = np.flip(np.eye(ksize), 0)
    var = ksize * ksize / 16.
    grid = np.repeat(np.arange(ksize)[:, np.newaxis], ksize, axis=-1)
    gaussian = np.exp(-(np.square(grid - center) + np.square(grid.T - center)) / (2. * var))
    kernel *= gaussian
    kernel /= np.sum(kernel)
    img = cv2.filter2D(img, -1, kernel)
    return img

# input batch*4*4 or batch*3*3
# output torch batch*3 x, y, z in radiant
# the rotation is in the sequence of x,y,z
def compute_euler_angles_from_rotation_matrices(rotation_matrices):
    # batch = rotation_matrices.shape[0]
    R = rotation_matrices
    sy = torch.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6
    singular = singular.float()

    x = torch.atan2(R[2, 1], R[2, 2])
    y = torch.atan2(-R[2, 0], sy)
    z = torch.atan2(R[1, 0], R[0, 0])

    xs = torch.atan2(-R[1, 2], R[1, 1])
    ys = torch.atan2(-R[2, 0], sy)
    zs = R[1, 0] * 0

    rotation_x = x * (1 - singular) + xs * singular
    rotation_y = y * (1 - singular) + ys * singular
    rotation_z = z * (1 - singular) + zs * singular

    return rotation_x, rotation_y, rotation_z
