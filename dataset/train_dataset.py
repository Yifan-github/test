import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from numpy.linalg import inv
from transforms3d.quaternions import *

from asset import name2database_set
from dataset.database import parse_seq_database_name, get_seq_database_pairs
from utils.base_utils import color_map_forward
from utils.dataset_utils import set_seed,compute_euler_angles_from_rotation_matrices


class ExampleCorrDataset(Dataset):
    default_cfg={
        "database_set_names": 'scannet_example',
        "database_pair_name": 'loftr_train',
        "img_type": 'gray',
    }
    def __init__(self, cfg, is_train):
        self.cfg = {**self.default_cfg,**cfg}
        self.is_train = is_train
        database_names = name2database_set[self.cfg['database_set_names']]
        self.databases, self.database_pairs, self.pair_num, self.cum_nums = [], {}, 0, []
        for database_name in database_names:
            database = parse_seq_database_name(database_name)
            self.databases.append(database)
            pairs = get_seq_database_pairs(database, self.cfg['database_pair_name'])
            self.database_pairs[database_name] = pairs
            self.pair_num+=len(pairs)
            self.cum_nums.append(len(pairs))

        self.cum_nums = np.asarray(self.cum_nums)
        self.cum_nums = np.cumsum(self.cum_nums)

    def get_database_pair(self, index):
        database_index = np.searchsorted(self.cum_nums, index, 'right')
        pair_index = index - self.cum_nums[database_index]
        database = self.databases[database_index]
        pair = self.database_pairs[database.database_name][pair_index]
        return database, pair

    def __getitem__(self, index):
        set_seed(index, self.is_train)
        database, pair = self.get_database_pair(index)
        id0, id1 = pair
        img0, img1 = database.get_image(id0), database.get_image(id1)
        pose0, pose1 = database.get_pose(id0), database.get_pose(id1)
        K0, K1 = database.get_K(id0), database.get_K(id1)
        depth_K0, depth_K1 = database.get_depth_K(id0), database.get_depth_K(id1)
        depth0, depth1 = database.get_depth(id0), database.get_depth(id1)

        if self.cfg['img_type']=='gray':
            img0 = cv2.cvtColor(img0, cv2.COLOR_RGB2GRAY)[None,:,:]
            img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)[None,:,:]
        elif self.cfg['img_type']=='rgb':
            pass
        else:
            raise NotImplementedError

        # read and compute relative poses
        R_0to1 = torch.tensor(np.matmul(pose1[0:3, 0:3], inv(pose0[0:3, 0:3])))
        R_1to0 = R_0to1.inverse()
        angle_x, angle_y, angle_z = compute_euler_angles_from_rotation_matrices(R_0to1)

        img0 = torch.from_numpy(color_map_forward(img0))
        img1 = torch.from_numpy(color_map_forward(img1))
        return {
            'img0': img0, 'img1': img1,
            'K0': torch.from_numpy(K0.astype(np.float32)),
            'K1': torch.from_numpy(K1.astype(np.float32)),
            'depth_K0': torch.from_numpy(depth_K0.astype(np.float32)),
            'depth_K1': torch.from_numpy(depth_K1.astype(np.float32)),
            'pose0': torch.from_numpy(pose0.astype(np.float32)),
            'pose1': torch.from_numpy(pose1.astype(np.float32)),
            'depth0': torch.from_numpy(depth0.astype(np.float32))[None,:,:],
            'depth1': torch.from_numpy(depth1.astype(np.float32))[None,:,:],
            'R_0to1': R_0to1,
            'R_1to0': R_1to0,
            'angle_x': angle_x,
            'angle_y': angle_y,
            'angle_z': angle_z
        }

    def __len__(self):
        return self.pair_num


class PoseRegDataset(Dataset):
    default_cfg = {
        "database_set_names": 'scannet_example',
        "database_pair_name": 'loftr_train',
        "img_type": 'gray',
    }
    def __init__(self, cfg, is_train, transforms=None):
        self.cfg = {**self.default_cfg,**cfg}
        self.is_train = is_train
        database_names = ['scannet_example/scene0000_00/240_320']
        self.databases, self.database_pairs, self.pair_num, self.cum_nums = [], {}, 0, []
        for database_name in database_names:
            database = parse_seq_database_name(database_name)#这里进入database，加载了k，w，h，和sequence路径
            self.databases.append(database)
            pairs = get_seq_database_pairs(database, self.cfg['database_pair_name'])#这里也进入database，加载了pairs
            self.database_pairs[database_name] = pairs
            self.pair_num+=len(pairs)
            self.cum_nums.append(len(pairs))

        self.cum_nums = np.asarray(self.cum_nums)
        self.cum_nums = np.cumsum(self.cum_nums)

        self.transforms = transforms

    def get_database_pair(self, index):
        database_index = np.searchsorted(self.cum_nums, index, 'right')
        pair_index = index - self.cum_nums[database_index]
        database = self.databases[database_index]
        pair = self.database_pairs[database.database_name][pair_index]
        return database, pair

    def __getitem__(self, index):
        set_seed(index, self.is_train)
        database, pair = self.get_database_pair(index)
        id0, id1 = pair
        img0, img1 = database.get_image(id0), database.get_image(id1)
        pose0, pose1 = database.get_pose(id0), database.get_pose(id1)
        K0, K1 = database.get_K(id0), database.get_K(id1)
        depth_K0, depth_K1 = database.get_depth_K(id0), database.get_depth_K(id1)
        depth0, depth1 = database.get_depth(id0), database.get_depth(id1)

        if self.cfg['img_type']=='gray':
            img0 = cv2.cvtColor(img0, cv2.COLOR_RGB2GRAY)[None,:,:]
            img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)[None,:,:]
        elif self.cfg['img_type']=='rgb':
            img0 = cv2.cvtColor(img0, cv2.cv2.COLOR_BGR2RGB)
            img1 = cv2.cvtColor(img1, cv2.cv2.COLOR_BGR2RGB)
        else:
            raise NotImplementedError

        # read and compute relative poses
        R_oto1 = np.matmul(pose1[0:3, 0:3], inv(pose0[0:3, 0:3]))
        quaternion = mat2quat(R_oto1)
        t_0to1 = pose1[:, -1] - R_oto1 @ pose0[:, -1]

        R_0to1 = torch.tensor(R_oto1)
        R_1to0 = R_0to1.inverse()
        angle_x, angle_y, angle_z = compute_euler_angles_from_rotation_matrices(R_0to1)

        img0 = Image.fromarray(img0)
        img1 = Image.fromarray(img1)
        if self.transforms is not None:
            img0 = self.transforms(img0)
            img1 = self.transforms(img1)

        img0 = np.array(img0)
        img1 = np.array(img1)

        img0 = torch.from_numpy(color_map_forward(img0))
        img1 = torch.from_numpy(color_map_forward(img1))


        return {
            'img0': img0, 'img1': img1,
            'K0': torch.from_numpy(K0.astype(np.float32)),
            'K1': torch.from_numpy(K1.astype(np.float32)),
            'depth_K0': torch.from_numpy(depth_K0.astype(np.float32)),
            'depth_K1': torch.from_numpy(depth_K1.astype(np.float32)),
            'pose0': torch.from_numpy(pose0.astype(np.float32)),
            'pose1': torch.from_numpy(pose1.astype(np.float32)),
            'depth0': torch.from_numpy(depth0.astype(np.float32))[None,:,:],
            'depth1': torch.from_numpy(depth1.astype(np.float32))[None,:,:],
            'R_0to1': R_0to1,
            'R_1to0': R_1to0,
            'quaternion': torch.from_numpy(quaternion.astype(np.float32)),
            't_0to1': torch.from_numpy(t_0to1.astype(np.float32))
        }

    def __len__(self):
        return self.pair_num
