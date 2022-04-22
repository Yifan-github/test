import numpy as np
import random
from pathlib import Path

from skimage.io import imsave

from dataset.database import parse_pair_database_name, parse_seq_database_name, get_seq_database_pairs
from dataset.name2dataset import name2dataset
from utils.base_utils import compute_F, pose_relative, interpolate_image_points, pts_to_hpts, transform_points_pose, \
    pose_inverse, project_points, to_numpy, color_map_backward
from utils.draw_utils import draw_epipolar_lines, concat_images_list, draw_keypoints


def validate_pair_database(database_name):
    database = parse_pair_database_name(database_name)
    pair_ids = database.get_pair_ids()
    random.seed(1234)
    random.shuffle(pair_ids)
    Path('data/validate').mkdir(parents=True, exist_ok=True)
    for pi, pair_id in enumerate(pair_ids[:10]):
        id0, id1 = database.get_image_id(pair_id)
        K0 = database.get_K(id0)
        K1 = database.get_K(id1)
        pose = database.get_pair_pose(pair_id)
        F = compute_F(K0,K1,pose[:3,:3],pose[:3,3:])
        img0, img1 = database.get_image(id0), database.get_image(id1)
        imsave(f'data/validate/{pi}.jpg', concat_images_list(*draw_epipolar_lines(F, img0, img1, 20)))

def validate_depth(img0,img1,depth0,depth_K0,pose0,pose1,K0,K1):
        # depth1 = database.get_depth(id1)
        h0, w0 = depth0.shape
        pn = 32
        pts2d = np.random.uniform(0,1,[pn,2])*np.asarray([[w0,h0]])
        pts_depth = interpolate_image_points(depth0, pts2d) # pn
        pts2d = pts_depth[:,None] * pts_to_hpts(pts2d)
        pts3d = pts2d @ np.linalg.inv(depth_K0).T
        pts3d = transform_points_pose(pts3d, pose_inverse(pose0))

        pts2d1, _ =project_points(pts3d,pose1,K1)
        pts2d0, _ =project_points(pts3d,pose0,K0)
        colors=np.random.randint(0,255,[pn,3])
        return concat_images_list(draw_keypoints(img0, pts2d0, colors, radius=5), draw_keypoints(img1, pts2d1, colors, radius=5))


def validate_seq_database(database_name, pair_name='exhaustive'):
    database = parse_seq_database_name(database_name)
    pairs = get_seq_database_pairs(database, pair_name)
    random.seed(1234)
    np.random.seed(1234)
    random.shuffle(pairs)
    for pi, pair in enumerate(pairs[:10]):
        id0, id1 = pair
        K0 = database.get_K(id0)
        K1 = database.get_K(id1)
        pose0, pose1 = database.get_pose(id0), database.get_pose(id1)
        pose = pose_relative(pose0, pose1)
        F = compute_F(K0,K1,pose[:3,:3],pose[:3,3:])
        img0, img1 = database.get_image(id0), database.get_image(id1)
        imsave(f'data/validate/{pi}.jpg', concat_images_list(*draw_epipolar_lines(F, img0, img1, 20)))

        # check depth
        depth0 = database.get_depth(id0)
        depth_K0 = database.get_depth_K(id0)
        imsave(f'data/validate/{pi}-depth.jpg',validate_depth(img0,img1,depth0,depth_K0,pose0,pose1,K0,K1))


def validate_example_dataset():
    cfg={}
    dataset = name2dataset['example'](cfg,True)
    for k in range(10):
        data = dataset[k]
        data = to_numpy(data)
        img0 = color_map_backward(np.repeat(data['img0'][0,:,:,None],3,2))
        img1 = color_map_backward(np.repeat(data['img1'][0,:,:,None],3,2))
        img = validate_depth(img0, img1, data['depth0'][0], data['depth_K0'], data['pose0'], data['pose1'], data['K0'], data['K1'])
        imsave(f'data/validate/{k}-depth.jpg', img)
