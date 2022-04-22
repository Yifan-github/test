import abc

import numpy as np
import torch
from cv2 import cv2

from network.superglue import SuperPoint, SuperGlue
from utils.base_utils import color_map_forward


class CorrEstimator(abc.ABC):
    @abc.abstractmethod
    def estimate(self, img0, img1, **kwargs):
        pass

class SuperGlueEstimator(CorrEstimator):
    default_cfg={
        'sp_cfg': {},
        'sg_cfg': {},
    }
    def __init__(self, cfg):
        self.cfg={**self.default_cfg, **cfg}
        self.sp=SuperPoint(self.cfg['sp_cfg']).cuda().eval()
        self.sg=SuperGlue(self.cfg['sg_cfg']).cuda().eval()

    def estimate(self, img0, img1, **kwargs):
        # to gray scale
        if len(img0.shape)==3 and img0.shape[2]==3:
            img0 = cv2.cvtColor(img0,cv2.COLOR_RGB2GRAY)
        if len(img1.shape)==3 and img1.shape[2]==3:
            img1 = cv2.cvtColor(img1,cv2.COLOR_RGB2GRAY)


        img0 = torch.from_numpy(color_map_forward(img0)[None,None]).cuda()
        img1 = torch.from_numpy(color_map_forward(img1)[None,None]).cuda()

        with torch.no_grad():
            results0 = self.sp({'image':img0})
            results1 = self.sp({'image':img1})

        data = {}
        data = {**data, **{k + '0': torch.stack(v,0) for k, v in results0.items()}}
        data = {**data, **{k + '1': torch.stack(v,0) for k, v in results1.items()}}
        data = {'image0': img0, 'image1': img1, **data}
        with torch.no_grad():
            results = self.sg(data)
        kps0, kps1 = data['keypoints0'].cpu().numpy()[0], data['keypoints1'].cpu().numpy()[0]
        matches0 = results['matches0'].cpu().numpy()[0]

        valid_mask = matches0!=-1
        matches = np.concatenate([kps0[valid_mask],kps1[matches0[valid_mask]]],1)
        return matches

name2corr_estimator={
    'superglue': SuperGlueEstimator
}