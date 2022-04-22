import argparse

from eval import PoseEvaluator
from train.trainer import Trainer
from train_pose.trainer_pose import TrainerPose
from utils.base_utils import load_cfg
from validation.validate_func import validate_pair_database, validate_seq_database, validate_example_dataset

parser = argparse.ArgumentParser()
parser.add_argument('--action', type=str, required=True)
parser.add_argument('--cfg', type=str, default='')
parser.add_argument('--database_name', type=str, default='scannet_test')
parser.add_argument('--database_type', type=str, default='pair')

parser.add_argument('--pair_name', type=str, default='loftr')

parser.add_argument('--dataset_path', type=str, default='/nfs/volume-315-5/zhaoyifan/projects/PoseCorr-main/data/scannet_seq_dataset/scene0000_00')
args = parser.parse_args()

if args.action=='val_pair':
    validate_pair_database(args.database_name)
elif args.action=='eval':
    PoseEvaluator(args.cfg, args.database_name, args.database_type).eval()
elif args.action=='val_seq':
    validate_seq_database(args.database_name, args.pair_name)
elif args.action=='val_example_dataset':
    validate_example_dataset()
elif args.action=='train':
    Trainer(load_cfg(args.cfg)).run()

elif args.action=='train_pose':
    TrainerPose(load_cfg(args.cfg)).run()
else:
    raise NotImplementedError