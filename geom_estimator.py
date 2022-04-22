import numpy as np
import cv2

class RANSACEstimator:
    default_cfg={
        "thresh": 1.,
        "conf": 0.99999
    }

    def __init__(self, cfg):
        self.cfg={**self.default_cfg, **cfg}

    def estimate(self, matches, K0, K1):
        kpts0, kpts1 = matches[:,:2], matches[:,2:]
        if len(kpts0) < 5:
            R, t, inlier_mask = np.eye(3), np.zeros(3,np.float32), np.zeros(matches.shape[0],np.bool_)
            return np.concatenate([R,t[:,None]],1), inlier_mask

        f_mean = np.mean([K0[0, 0], K1[1, 1], K0[0, 0], K1[1, 1]])
        norm_thresh = self.cfg['thresh'] / f_mean

        kpts0 = (kpts0 - K0[[0, 1], [2, 2]][None]) / K0[[0, 1], [0, 1]][None]
        kpts1 = (kpts1 - K1[[0, 1], [2, 2]][None]) / K1[[0, 1], [0, 1]][None]

        E, mask = cv2.findEssentialMat(
            kpts0, kpts1, np.eye(3), threshold=norm_thresh, prob=self.cfg['conf'],
            method=cv2.RANSAC)

        assert E is not None

        best_num_inliers = 0
        ret = None
        for _E in np.split(E, len(E) / 3):
            n, R, t, _ = cv2.recoverPose(
                _E, kpts0, kpts1, np.eye(3), 1e9, mask=mask)
            if n > best_num_inliers:
                best_num_inliers = n
                ret = (R, t[:, 0], mask.ravel() > 0)
        if ret is not None:
            R, t, inlier_mask = ret
        else:
            R, t, inlier_mask = np.eye(3), np.zeros(3,np.float32), np.zeros(matches.shape[0],np.bool_)
        return np.concatenate([R,t[:,None]],1), inlier_mask

name2geom_estimator={
    'ransac': RANSACEstimator
}