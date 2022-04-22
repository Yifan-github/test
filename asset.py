import os

import numpy as np

SCANNET_ROOT='data/scannet_dataset'
SCANNET_SEQUENCE_ROOT='data/scannet_seq_dataset'

POSE_CACHE='data/pose_cache'
MATCH_CACHE='data/match_cache'
VIS_CACHE='data/vis_cache'

# if os.path.exists(f'{SCANNET_SEQUENCE_ROOT}/scannet_indices/intrinsics.npz'):
#     SCANNET_SEQUENCE_Ks=np.load(f'{SCANNET_SEQUENCE_ROOT}/scannet_indices/intrinsics.npz')
# else:
#     SCANNET_SEQUENCE_Ks={}

# scannet_example_sets=['scannet_example/scene0000_00/480_640','scannet_example/scene0000_01/480_640']

scannet_example_sets=['scannet_example/scene0000_00/240_320']

name2database_set={
    'scannet_example': scannet_example_sets
}