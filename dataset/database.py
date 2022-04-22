import abc
import os

import cv2
import numpy as np
from skimage.io import imread

from asset import SCANNET_ROOT, SCANNET_SEQUENCE_ROOT
from utils.base_utils import pose_inverse


class SequenceBaseDatabase(abc.ABC):
    def __init__(self, database_name):
        self.database_name = database_name

    @abc.abstractmethod
    def get_image(self, img_id):
        pass

    @abc.abstractmethod
    def get_K(self, img_id):
        pass

    def get_depth_K(self, img_id):
        return self.get_K(img_id)

    @abc.abstractmethod
    def get_pose(self, img_id):
        pass

    @abc.abstractmethod
    def get_img_ids(self):
        pass

    @abc.abstractmethod
    def get_depth(self,img_id):
        pass

    @abc.abstractmethod
    def get_mask(self,img_id):
        pass

class PairBaseDatabase(abc.ABC):
    def __init__(self, database_name):
        self.database_name = database_name

    @abc.abstractmethod
    def get_image(self, img_id):
        pass

    @abc.abstractmethod
    def get_K(self, img_id):
        pass

    @abc.abstractmethod
    def get_depth(self,img_id):
        pass

    @abc.abstractmethod
    def get_mask(self,img_id):
        pass

    @abc.abstractmethod
    def get_pair_pose(self, pair_id):
        pass

    @abc.abstractmethod
    def get_pair_ids(self):
        pass

    @staticmethod
    def get_image_id(pair_id):
        return pair_id.split('+')

class ScannetTestPair(PairBaseDatabase):
    def __init__(self, database_name):
        super().__init__(database_name)
        self.pair_ids, self.pair_poses, self.image_Ks = self.parse_pairs()

    @staticmethod
    def parse_pairs():
        fn = 'assets/scannet_test_pairs_with_gt.txt'

        def get_scene_frame(image_name):
            scene_id_str = image_name.split('/')[1][5:]
            frame_id_str = image_name.split('/')[-1][6:12]
            return scene_id_str, frame_id_str

        pair_ids, pair_poses, image_Ks = [], {}, {}
        with open(fn, 'r') as f:
            for line in f.readlines():
                items = line.split(' ')
                scene_id0, frame_id0 = get_scene_frame(items[0])
                scene_id1, frame_id1 = get_scene_frame(items[1])

                K0 = np.array(items[4:13]).astype(float).reshape(3, 3)
                K1 = np.array(items[13:22]).astype(float).reshape(3, 3)
                pose = np.array(items[22:]).astype(float).reshape(4, 4)

                img_id0 = '-'.join([scene_id0,frame_id0])
                img_id1 = '-'.join([scene_id1,frame_id1])
                pair_id = '+'.join([img_id0, img_id1])
                pair_ids.append(pair_id)
                pair_poses[pair_id] = pose[:3,:4]
                image_Ks[img_id0] = K0
                image_Ks[img_id1] = K1

        return pair_ids, pair_poses, image_Ks

    def get_image(self, img_id):
        scen_id, frame_id = img_id.split('-')
        frame_id = int(frame_id)
        img = imread(f'{SCANNET_ROOT}/scans_test/scene{scen_id}/color/{frame_id}.jpg')
        return img

    def get_K(self, img_id):
        return np.copy(self.image_Ks[img_id])

    def get_depth(self, img_id):
        raise NotImplementedError

    def get_mask(self, img_id):
        raise NotImplementedError

    def get_pair_pose(self, pair_id):
        return np.copy(self.pair_poses[pair_id])

    def get_pair_ids(self):
        return self.pair_ids.copy()

def read_scannet_depth(path):
    depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    depth = depth.astype(np.float32) / 1000
    return depth

class ScannetExample(SequenceBaseDatabase):
    def __init__(self, database_name):
        super().__init__(database_name)
        _, self.seq_name, img_size = database_name.split('/') # scannet_example/seq_name
        self.root = f'{SCANNET_SEQUENCE_ROOT}/{self.seq_name}'
        self.img_num = len(os.listdir(f'{self.root}/color'))
        self.K = np.loadtxt(f'{self.root}/intrinsic/intrinsic_color.txt')[:3,:3].astype(np.float32)
        self.depth_K = np.loadtxt(f'{self.root}/intrinsic/intrinsic_depth.txt')[:3,:3].astype(np.float32)
        self.h, self.w = [int(size) for size in img_size.split('_')]
        h_raw, w_raw = 968, 1296
        self.K = np.diag([self.w/w_raw, self.h/h_raw, 1.0]) @ self.K

    def get_image(self, img_id):
        img=imread(f'{self.root}/color/{img_id}.jpg')
        return cv2.resize(img,(self.w,self.h))

    def get_K(self, img_id):
        return self.K.copy()

    def get_depth_K(self, img_id):
        return self.depth_K.copy()

    def get_pose(self, img_id):
        pose=np.loadtxt(f'{self.root}/pose/{img_id}.txt').astype(np.float32)
        pose=pose[:3,:4]
        return pose_inverse(pose)

    def get_img_ids(self):
        return [str(k) for k in range(self.img_num)]

    def get_depth(self, img_id):
        return read_scannet_depth(f'{self.root}/depth/{img_id}.png')

    def get_mask(self, img_id):
        raise NotImplementedError


def parse_seq_database_name(database_name) -> SequenceBaseDatabase:
    root_name = database_name.split('/')[0]
    root_name2database={
        'scannet_example': ScannetExample,
    }
    return root_name2database[root_name](database_name)

def get_seq_database_pairs(database: SequenceBaseDatabase, pair_name):
    if pair_name.startswith('loftr'):
        if isinstance(database, ScannetExample):
            data = np.load(f'{SCANNET_SEQUENCE_ROOT}/scannet_indices/scene_data/train/{database.seq_name}.npz')
            if pair_name=='loftr_train':
                return data['name'][:, 2:4].astype(np.str)
            elif pair_name=='loftr_val':
                np.random.seed(1234)
                ids=np.arange(data['name'].shape[0])
                np.random.shuffle(ids)
                return data['name'][ids[:100],2:4]
        else:
            raise NotImplementedError
    elif pair_name=='exhaustive':
        img_ids = database.get_img_ids()
        pairs=[]
        for i in range(len(img_ids)):
            for j in range(i+1,len(img_ids)):
                pairs.append([img_ids[i], img_ids[j]])
        return np.asarray(pairs)
    else:
        raise NotImplementedError

def parse_pair_database_name(database_name: str) -> PairBaseDatabase:
    root_name = database_name.split('/')[0]
    root_name2database={
        'scannet_test': ScannetTestPair,
    }
    return root_name2database[root_name](database_name)
