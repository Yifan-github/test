import torch
import torch.nn as nn

class MlpBottleneck(nn.Module):
    def __init__(self, in_planes, planes):
        super(MlpBottleneck, self).__init__()

        self.mlp1 = nn.Conv1d(128, 128, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(128)

        self.mlp2 = nn.Conv1d(128, 128, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm1d(128)

        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        identity = x
        # if self.downsample is not None:
        #     identity = self.downsample(x)

        out = self.mlp1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.mlp2(out)
        out = self.bn2(out)
        # out = self.relu(out)

        # 这里先加残差还是先relu？
        out += identity
        out = self.relu(out)

        return out

# input : batch * n * 5
# output : batch * n * f

# linear : 2048 , zdim , out_dim
# 2048 512 3

class PoseDecoder(nn.Module):
    def __init__(self, config):
        super(PoseDecoder, self).__init__()
        self.numlayer = 12  # 12
        self.nchannel = 128 # 128
        block = MlpBottleneck
        self.out_rotation_mode = "Quaternion"
        self.out_dim_r = 4
        self.out_dim_t = 3

        self.mlp1 = nn.Conv1d(5, 128, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(128)
        self.relu = nn.ReLU(inplace=False)

        self.layer1 = self._make_layer(block, self.nchannel)
        self.layer2 = self._make_layer(block, self.nchannel)
        self.layer3 = self._make_layer(block, self.nchannel)
        self.layer4 = self._make_layer(block, self.nchannel)
        self.layer5 = self._make_layer(block, self.nchannel)
        self.layer6 = self._make_layer(block, self.nchannel)
        self.layer7 = self._make_layer(block, self.nchannel)
        self.layer8 = self._make_layer(block, self.nchannel)

        self.linear = nn.Linear(in_features=128, out_features=self.out_dim_r, bias=True)
        self.linear2 = nn.Linear(in_features=128, out_features=self.out_dim_t, bias=True)

    def _make_layer(self, block, zim_channel):
        layers = []
        layers.append(block(zim_channel, zim_channel))
        return nn.Sequential(*layers)

    def forward(self, x):
        batch = x.shape[0]
        out = self.mlp1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)
        out = torch.mean(out, dim=2)
        out = out.view(out.size(0), -1)
        out_r = self.linear(out)
        out_t = self.linear2(out)
        return out_r, out_t
