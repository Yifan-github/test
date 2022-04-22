from pathlib import Path

import numpy as np
import torch
from skimage.io import imsave

from geom_estimator import RANSACEstimator
from network.loss import Loss
from utils.base_utils import pose_relative, color_map_backward
from utils.draw_utils import draw_correspondence
from utils.metrics_utils import evaluate_R_t, pose_metrics

class PoseErrors(Loss):
    default_cfg={}
    def __init__(self, cfg):
        super().__init__(['pose_errs'])
        if (type(cfg)!=dict):
            cfg = vars(cfg)
        self.cfg={**self.default_cfg,**cfg}
        self.estimator = RANSACEstimator({})

    def __call__(self, data_pr, data_gt, step, **kwargs):
        kps0 = data_pr['keypoints0'] # b,k,2
        kps1 = data_pr['keypoints1'] # b,t,2
        K0 = data_gt['K0'] # b,3,3
        K1 = data_gt['K1'] # b,3,3
        pose0 = data_gt['pose0']
        pose1 = data_gt['pose1']
        matches = data_pr['matches0']  # b,k

        kps0, kps1, matches = kps0.cpu().numpy(), kps1.cpu().numpy(), matches.cpu().numpy()
        K0, K1, pose0, pose1 = K0.cpu().numpy(), K1.cpu().numpy(), pose0.cpu().numpy(), pose1.cpu().numpy()
        b = kps0.shape[0]
        pose_errs=[]
        for bi in range(b):
            mask = matches[bi]!=-1
            kps0_ = kps0[bi][mask]
            kps1_ = kps1[bi][matches[bi][mask]]
            pose_pr, _ = self.estimator.estimate(np.concatenate([kps0_,kps1_],1), K0[bi], K1[bi])
            pose_gt = pose_relative(pose0[bi], pose1[bi])
            R_gt, t_gt, R_pr, t_pr = pose_gt[:3, :3], pose_gt[:3, 3], pose_pr[:3, :3], pose_pr[:3, 3]
            R_err, t_err = evaluate_R_t(R_gt, t_gt, R_pr, t_pr)
            pose_errs.append(np.asarray([R_err,t_err]))
        pose_errs = torch.from_numpy(np.stack(pose_errs,0).astype(np.float32)) # b,2
        return {'pose_errs': pose_errs}

class VisCorr(Loss):
    default_cfg={
        'output_interval': 25,
    }
    def __init__(self, cfg):
        super().__init__([])
        if (type(cfg)!=dict):
            cfg = vars(cfg)
        self.cfg={**self.default_cfg,**cfg}

    def __call__(self, data_pr, data_gt, step, **kwargs):
        data_index=kwargs['data_index']
        model_name=kwargs['model_name']
        if data_index % self.cfg['output_interval']!=0:
            return {}

        output_root = kwargs['output_root'] if 'output_root' in kwargs else 'data/train_vis'
        Path(output_root).mkdir(parents=True, exist_ok=True)

        kps0 = data_pr['keypoints0'] # b,k,2
        kps1 = data_pr['keypoints1'] # b,t,2
        matches = data_pr['matches0']  # b,k
        img0 = data_gt['img0']
        img1 = data_gt['img1']

        img0 = color_map_backward(img0.permute(0,2,3,1).cpu().numpy())[0]
        img1 = color_map_backward(img1.permute(0,2,3,1).cpu().numpy())[0]

        kps0, kps1, matches = kps0.cpu().numpy(), kps1.cpu().numpy(), matches.cpu().numpy()
        bi = 0
        mask = matches[bi] != -1
        kps0_ = kps0[bi][mask]
        kps1_ = kps1[bi][matches[bi][mask]]
        imsave(f'{output_root}/{model_name}-{step}-{data_index}-corr.jpg', draw_correspondence(img0,img1,kps0_,kps1_,))
        return {}

name2metrics={
    'pose_errs': PoseErrors,
    'vis_corr': VisCorr,
}

def pose_auc_5(results):
    pose_errs = results['pose_errs'] # b,2
    _, names, aucs = pose_metrics(pose_errs)
    return aucs[0]

name2key_metrics={
    'pose_auc_5': pose_auc_5,
}