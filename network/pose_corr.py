import torch.nn as nn
import torch
from network.superglue import KeypointEncoder, AttentionalGNN, normalize_keypoints, log_optimal_transport, arange_like, \
    SuperPoint, simple_nms, top_k_keypoints, sample_descriptors


class SuperPointFixed(SuperPoint):
    def forward(self, data):
        """ Compute keypoints, scores, descriptors for image """
        # Shared Encoder
        x = self.relu(self.conv1a(data['image']))
        x = self.relu(self.conv1b(x))
        x = self.pool(x)
        x = self.relu(self.conv2a(x))
        x = self.relu(self.conv2b(x))
        x = self.pool(x)
        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool(x)
        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))

        # Compute the dense keypoint scores
        cPa = self.relu(self.convPa(x))
        scores = self.convPb(cPa)
        scores = torch.nn.functional.softmax(scores, 1)[:, :-1]
        b, _, h, w = scores.shape
        scores = scores.permute(0, 2, 3, 1).reshape(b, h, w, 8, 8)
        scores = scores.permute(0, 1, 3, 2, 4).reshape(b, h*8, w*8)
        scores = simple_nms(scores, self.config['nms_radius'])

        # Extract keypoints
        keypoints = [torch.nonzero(s > 0) for s in scores]
        scores = [s[tuple(k.t())] for s, k in zip(scores, keypoints)]

        # Keep the k keypoints with highest score
        keypoints, scores = list(zip(*[top_k_keypoints(k, s, self.config['max_keypoints']) for k, s in zip(keypoints, scores)]))

        # Convert (h, w) to (x, y)
        keypoints = [torch.flip(k, [1]).float() for k in keypoints]

        # Compute the dense descriptors
        cDa = self.relu(self.convDa(x))
        descriptors = self.convDb(cDa)
        descriptors = torch.nn.functional.normalize(descriptors, p=2, dim=1)

        # Extract descriptors
        keypoints = torch.stack(keypoints,0)
        scores = torch.stack(scores,0)
        descriptors = sample_descriptors(keypoints, descriptors, 8)
        return {
            'keypoints': keypoints,
            'scores': scores,
            'descriptors': descriptors,
        }

class SuperGlueExample(nn.Module):
    default_config = {
        'descriptor_dim': 256,
        'weights': 'indoor',
        'keypoint_encoder': [32, 128, 256],
        'GNN_layers': ['self', 'cross'] * 2,
        'sinkhorn_iterations': 10,
        'match_threshold': 0.1,
        'kps_max': 256,
    }
    def __init__(self, config):
        super().__init__()
        self.config = {**self.default_config, **config}

        # not require gradient for superpoint
        self.superpoint = SuperPointFixed({'max_keypoints': self.config['kps_max']})
        for para in self.superpoint.parameters():
            para.requires_grad = False

        self.kenc = KeypointEncoder(
            self.config['descriptor_dim'], self.config['keypoint_encoder'])

        self.gnn = AttentionalGNN(
            self.config['descriptor_dim'], self.config['GNN_layers'])

        self.final_proj = nn.Conv1d(
            self.config['descriptor_dim'], self.config['descriptor_dim'],
            kernel_size=1, bias=True)

        bin_score = torch.nn.Parameter(torch.tensor(1.))
        self.register_parameter('bin_score', bin_score)

    def predict(self, inputs):
        """Run SuperGlue on a pair of keypoints and descriptors"""
        desc0, desc1 = inputs['descriptors0'], inputs['descriptors1']
        kpts0, kpts1 = inputs['keypoints0'], inputs['keypoints1']

        if kpts0.shape[1] == 0 or kpts1.shape[1] == 0:  # no keypoints
            shape0, shape1 = kpts0.shape[:-1], kpts1.shape[:-1]
            return {
                'matches0': kpts0.new_full(shape0, -1, dtype=torch.int),
                'matches1': kpts1.new_full(shape1, -1, dtype=torch.int),
                'matching_scores0': kpts0.new_zeros(shape0),
                'matching_scores1': kpts1.new_zeros(shape1),
            }

        # Keypoint normalization.
        kpts0 = normalize_keypoints(kpts0, inputs['image0'].shape)
        kpts1 = normalize_keypoints(kpts1, inputs['image1'].shape)

        # Keypoint MLP encoder.
        desc0 = desc0 + self.kenc(kpts0, inputs['scores0'])
        desc1 = desc1 + self.kenc(kpts1, inputs['scores1'])

        # Multi-layer Transformer network.
        desc0, desc1 = self.gnn(desc0, desc1)

        # Final MLP projection.
        mdesc0, mdesc1 = self.final_proj(desc0), self.final_proj(desc1)

        # Compute matching descriptor distance.
        scores = torch.einsum('bdn,bdm->bnm', mdesc0, mdesc1)
        scores = scores / self.config['descriptor_dim']**.5

        # Run the optimal transport.
        scores = log_optimal_transport(
            scores, self.bin_score,
            iters=self.config['sinkhorn_iterations'])

        # Get the matches with score above "match_threshold".
        max0, max1 = scores[:, :-1, :-1].max(2), scores[:, :-1, :-1].max(1)
        indices0, indices1 = max0.indices, max1.indices
        mutual0 = arange_like(indices0, 1)[None] == indices1.gather(1, indices0)
        mutual1 = arange_like(indices1, 1)[None] == indices0.gather(1, indices1)
        zero = scores.new_tensor(0)
        mscores0 = torch.where(mutual0, max0.values.exp(), zero)
        mscores1 = torch.where(mutual1, mscores0.gather(1, indices1), zero)
        valid0 = mutual0 & (mscores0 > self.config['match_threshold'])
        valid1 = mutual1 & valid0.gather(1, indices1)
        indices0 = torch.where(valid0, indices0, indices0.new_tensor(-1))
        indices1 = torch.where(valid1, indices1, indices1.new_tensor(-1))

        return {
            'matches0': indices0, # use -1 for invalid match
            'matches1': indices1, # use -1 for invalid match
            'matching_scores0': mscores0,
            'matching_scores1': mscores1,
            'scores': scores, # b,k,t
        }

    def forward(self, data):
        img0 = data['img0']
        img1 = data['img1']

        self.superpoint.eval()
        with torch.no_grad():
            predict0 = self.superpoint({'image': img0})
            predict1 = self.superpoint({'image': img1})

        inputs={
            'keypoints0': predict0['keypoints'],
            'descriptors0': predict0['descriptors'],
            'scores0': predict0['scores'],
            'keypoints1': predict1['keypoints'],
            'descriptors1': predict1['descriptors'],
            'scores1': predict1['scores'],
            'image0': img0,
            'image1': img1,
        }
        outputs = self.predict(inputs)
        return {**outputs,
            'keypoints0': inputs['keypoints0'],
            'keypoints1': inputs['keypoints1'],
        }