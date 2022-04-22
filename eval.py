from pathlib import Path

from skimage.io import imsave
from tqdm import tqdm

from asset import POSE_CACHE, MATCH_CACHE, VIS_CACHE
from corr_estimator import name2corr_estimator
from dataset.database import parse_pair_database_name, parse_seq_database_name, get_seq_database_pairs, PairBaseDatabase
from geom_estimator import name2geom_estimator
from utils.base_utils import load_cfg, save_pickle, read_pickle

import numpy as np

from utils.draw_utils import draw_correspondence, get_colors_gt_pr
from utils.metrics_utils import evaluate_R_t, pose_metrics


class PoseEvaluator:
    def __init__(self, eval_cfg, database_name, database_type, database_pair_type=None):
        cfg = load_cfg(eval_cfg)
        self.name = cfg['name']
        corr_estimator_cfg = load_cfg(cfg['corr_estimator'])
        geom_estimator_cfg = load_cfg(cfg['geom_estimator'])
        self.corr_estimator = name2corr_estimator[corr_estimator_cfg['type']](corr_estimator_cfg)
        self.geom_estimator = name2geom_estimator[geom_estimator_cfg['type']](geom_estimator_cfg)

        self.database_type = database_type
        if self.database_type == 'pair':
            self.eval_database = parse_pair_database_name(database_name)
        elif self.database_type == 'seq':
            self.eval_database = parse_seq_database_name(database_name)
            self.eval_pairs = get_seq_database_pairs(self.eval_database, database_pair_type)
        else:
            raise NotImplementedError

    def eval(self):
        if self.database_type=='pair':
            assert(isinstance(self.eval_database, PairBaseDatabase))
            (Path(f'{POSE_CACHE}') / self.name / self.eval_database.database_name).mkdir(exist_ok=True, parents=True)
            (Path(f'{MATCH_CACHE}') / self.name / self.eval_database.database_name).mkdir(exist_ok=True, parents=True)
            (Path(f'{VIS_CACHE}') / self.name / self.eval_database.database_name).mkdir(exist_ok=True, parents=True)

            R_errs, t_errs = [], []
            for pair_id in tqdm(self.eval_database.get_pair_ids()):
                img_id0, img_id1 = self.eval_database.get_image_id(pair_id)
                pose_fn = Path(f'{POSE_CACHE}') / self.name / self.eval_database.database_name / f'{img_id0}-{img_id1}.pkl'
                match_fn = Path(f'{MATCH_CACHE}') / self.name / self.eval_database.database_name / f'{img_id0}-{img_id1}.pkl'

                if not match_fn.exists():
                    img0, img1 = self.eval_database.get_image(img_id0), self.eval_database.get_image(img_id1)
                    matches = self.corr_estimator.estimate(img0, img1)
                    save_pickle(matches, str(match_fn))
                else:
                    matches = read_pickle(str(match_fn))

                if not pose_fn.exists():
                    K0, K1 = self.eval_database.get_K(img_id0), self.eval_database.get_K(img_id1)
                    pose_pr, inlier_mask = self.geom_estimator.estimate(matches, K0, K1)
                    save_pickle((pose_pr, inlier_mask),str(pose_fn))
                else:
                    pose_pr, inlier_mask = read_pickle(str(pose_fn))

                # comptue pose errors
                pose_gt = self.eval_database.get_pair_pose(pair_id)
                R_gt, t_gt, R_pr, t_pr = pose_gt[:3,:3], pose_gt[:3,3], pose_pr[:3,:3], pose_pr[:3,3]
                R_err, t_err = evaluate_R_t(R_gt, t_gt, R_pr, t_pr)
                R_errs.append(R_err); t_errs.append(t_err)

                # visualize
                img0, img1 = self.eval_database.get_image(img_id0), self.eval_database.get_image(img_id1)
                colors = get_colors_gt_pr(inlier_mask)
                corr_img = draw_correspondence(img0, img1, matches[:,:2], matches[:,2:], colors=colors)
                fn = Path(f'{VIS_CACHE}/{self.name}/{self.eval_database.database_name}/{img_id0}-{img_id1}-{R_err:.2f}-{t_err:.2f}.jpg')
                imsave(str(fn), corr_img)
        else:
            raise NotImplementedError

        Rt_errs = np.stack([np.stack(R_errs), np.stack(t_errs)], 1)
        print(pose_metrics(Rt_errs)[0])