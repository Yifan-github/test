import numpy as np
import torch
import torch.nn.functional as F
from network.operator import normalize_coords, denormalize_coords


class Loss:
    def __init__(self, keys):
        """
        keys are used in multi-gpu model, DummyLoss in train_tools.py
        :param keys: the output keys of the dict
        """
        self.keys=keys

    def __call__(self, data_pr, data_gt, step, **kwargs):
        pass

class ClsCorrLoss(Loss):
    default_cfg={
        "match_pixel_thresh": 8,
        "unmatch_pixel_thresh": 15,
        "depth_consistency_ratio": 0.2,
    }
    def __init__(self,cfg):
        self.cfg={**self.default_cfg,**cfg}
        super().__init__(['loss_cls'])

    @staticmethod
    def get_gt_mask(kps0, kps1, depth0, depth_K0, pose0, K1, pose1, h, w,
                    match_pixel_thresh=8, unmatch_pixel_thresh=15, depth_interpolate_mode='nearest',
                    eps=1e-3, depth_consistency_ratio=0.2):
        """
        :param kps0:      b,k,2
        :param kps1:      b,t,2
        :param depth0:    b,1,hd,wd
        :param depth_K0:  b,3,3
        :param pose0:     b,3,4
        :param K1:        b,3,3
        :param pose1:     b,3,3
        :param h:         int
        :param w:         int
        :param depth_interpolate_mode:
        :param eps:
        :param match_pixel_thresh:
        :param unmatch_pixel_thresh:
        :param depth_consistency_ratio:
        :return:
        """
        kps0_ = normalize_coords(kps0, h, w) # b,k,2
        kps0_depth = F.grid_sample(depth0,kps0_.unsqueeze(1),depth_interpolate_mode)[:,0,0,:] # b,k

        b, _, hd, wd = depth0.shape
        kps0_ = denormalize_coords(kps0_, hd, wd) # b,k,2, on depth map
        b, k, _ = kps0.shape
        ones = torch.ones([b,k,1],device=kps0.device,dtype=torch.float32)
        kps0_ = torch.cat([kps0_,ones],-1)          # b,k,3, homogeneous coodinates
        kps0_ = kps0_depth[:,:,None] * kps0_        # b,k,3
        depth_K0_inv = torch.inverse(depth_K0.cpu()).to(kps0.device)      # b,3,3
        kps3d = kps0_ @ depth_K0_inv.permute(0,2,1) # b,k,3, 3d coordinates in camera 0 coordinte

        # (b,k,3 - b,1,3) @ b,3,3
        kps3d = (kps3d - pose0[:,:3,3:].permute(0,2,1)) @ pose0[:,:3,:3] # coordinate in world coordinate
        kps3d = kps3d @ pose1[:,:3,:3].permute(0,2,1) + pose1[:,:3,3:].permute(0,2,1) # coorindate in camera1 coordinate
        kps3d = kps3d @ K1.permute(0,2,1) # b,k,3
        kps1_prj_depth = kps3d[:,:,2]     # b,k

        depth_valid_mask = kps1_prj_depth>eps # b,k
        kps1_prj_depth = torch.clip(kps1_prj_depth, min=1e-3) # prevent divide 0
        kps1_prj = kps3d[:,:,:2] / kps1_prj_depth[:,:,None] # b,k,2

        kps1_prj_norm = normalize_coords(kps1_prj, h, w)
        kps1_inter_depth = F.grid_sample(depth0,kps1_prj_norm.unsqueeze(1),depth_interpolate_mode)[:,0,0,:] # b,k
        consistent_mask = (kps1_inter_depth - kps1_prj_depth)/ kps1_inter_depth < depth_consistency_ratio

        dist2d = torch.norm(kps1_prj.unsqueeze(2) - kps1.unsqueeze(1),dim=-1) # b,k,t
        dist_mask = dist2d < match_pixel_thresh
        labels_indices = torch.argmin(dist2d,-1) # b,k
        pos_labels = torch.zeros_like(dist2d)
        pos_labels[torch.arange(b)[:,None], torch.arange(k)[None,:], labels_indices]=1.0 # b,k,t

        # mutual nearest
        t = kps1.shape[1]
        labels_indices_ = torch.argmin(dist2d,1) # b,t
        pos_labels_ = torch.zeros_like(dist2d) # b,k,t
        pos_labels_[torch.arange(b)[:,None], labels_indices_, torch.arange(t)[None,:]]=1.0 # b,k,t
        pos_labels = pos_labels * pos_labels_

        depth_valid_mask, consistent_mask, dist_mask = depth_valid_mask.float(), consistent_mask.float(), dist_mask.float()
        pos_labels = pos_labels * depth_valid_mask[:, :, None] * consistent_mask[:, :, None] * dist_mask

        # negative labels
        neg_labels = (dist2d>unmatch_pixel_thresh).float()
        neg_labels0 = (torch.sum(neg_labels,2) == t).float() # b,k
        neg_labels1 = (torch.sum(neg_labels,1) == k).float() # b,t
        return pos_labels, neg_labels0, neg_labels1

    @staticmethod
    def check_labels(kps0, kps1, img0, img1, pos_labels):
        """

        :param kps0:   b,k,2
        :param kps1:   b,t,2
        :param img0:   b,3,h,w
        :param img1:   b,3,h,w
        :param pos_labels: b,k,t
        :return:
        """
        from utils.base_utils import color_map_backward
        from utils.draw_utils import concat_images_list, draw_correspondence
        from skimage.io import imsave
        img0 = color_map_backward(img0.permute(0,2,3,1).cpu().numpy())
        img1 = color_map_backward(img1.permute(0,2,3,1).cpu().numpy())

        b, _, h, w = img0.shape
        k, t = kps0.shape[1],kps1.shape[1]
        imgs=[]
        kps0, kps1 = kps0.cpu().numpy(), kps1.cpu().numpy()
        pos_labels = pos_labels.cpu().numpy()
        for bi in range(b):
            ids0, ids1 = np.nonzero(pos_labels[bi])
            kps0_ = kps0[bi][ids0]
            kps1_ = kps1[bi][ids1]
            img = draw_correspondence(img0[bi],img1[bi],kps0_,kps1_)
            imgs.append(img)
        imsave('data/validate/labels.jpg',concat_images_list(*imgs))
        import ipdb; ipdb.set_trace()

    def __call__(self, data_pr, data_gt, step, *args, **kwargs):
        kps0, kps1 = data_pr['keypoints0'], data_pr['keypoints1'] # b,k,2

        pose0, pose1 = data_gt['pose0'], data_gt['pose1'] # b,3,4
        K0, K1 = data_gt['K0'], data_gt['K1'] # b,3,3
        depth_K0, depth_K1 = data_gt['depth_K0'], data_gt['depth_K1'] # b,3,3
        depth0, depth1 = data_gt['depth0'], data_gt['depth1'] # b,1,hd,wd

        b, _, h, w = data_gt['img0'].shape
        assert(h==data_gt['img1'].shape[2] and w==data_gt['img1'].shape[3]) # same size
        pos_labels, neg_labels0, neg_labels1 = self.get_gt_mask(
            kps0, kps1, depth0, depth_K1, pose0, K1, pose1, h, w,
            match_pixel_thresh=self.cfg['match_pixel_thresh'], unmatch_pixel_thresh=self.cfg['unmatch_pixel_thresh'],
            depth_consistency_ratio=self.cfg['depth_consistency_ratio']) # b,k,t

        # self.check_labels(kps0,kps1,data_gt['img0'],data_gt['img1'],pos_labels)

        scores = data_pr['scores'] # b,k+1,t+1
        k, t = kps0.shape[1], kps1.shape[1]
        pos_loss = -pos_labels * scores[:,:k,:t]
        pos_loss = torch.sum(pos_loss.flatten(1),1)/(torch.sum(pos_labels.flatten(1),1)+1e-4) # b

        scores_exp0 = torch.exp(scores[:,:k,t]) # b,k
        scores_exp0 = torch.clip(scores_exp0, max=1-1e-5)
        neg_loss0 = -neg_labels0 * torch.log(1 - scores_exp0)
        neg_loss0 = torch.sum(neg_loss0.flatten(1),1)/(torch.sum(neg_labels0.flatten(1),1)+1e-4) # b

        scores_exp1 = torch.exp(scores[:,k,:t]) # b,t
        scores_exp1 = torch.clip(scores_exp1, max=1-1e-5)
        neg_loss1 = -neg_labels1 * torch.log(1 - scores_exp1)
        neg_loss1 = torch.sum(neg_loss1.flatten(1),1)/(torch.sum(neg_labels1.flatten(1),1)+1e-4) # b
        neg_loss = (neg_loss0 + neg_loss1)/2

        # compute accuracy
        # matches0_pr = torch.argmax(scores[:,:k,:],2) # b,k
        # matches0_pr[matches0_pr==t]=-1 # dust bin class
        # matches0_gt = torch.argmax(pos_labels[:,:k,:],2) # b,k
        # matches0_gt[matches0_gt==t]=-1
        # precision = torch.mean(((matches0_gt==matches0_pr) & (matches0_pr!=-1)).float(),1) # b
        # recall = torch.mean(((matches0_gt==matches0_pr) & (matches0_gt!=-1)).float(),1) # b
        return {'loss_cls': (pos_loss+neg_loss)/2} # b

name2loss={
    'cls_corr': ClsCorrLoss,
}