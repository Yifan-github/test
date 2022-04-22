import os
import importlib
import argparse
import sys

import torch
import numpy as np
from torch.optim import Adam, SGD
from torch import optim
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.name2dataset import name2dataset
from network.loss import name2loss
from network.name2network import name2network
# from train.lr_common_manager import name2lr_manager
from network.metrics import name2metrics
from train.train_tools import to_cuda, Logger
from train.train_valid import ValidationEvaluator
from utils.dataset_utils import dummy_collate_fn, simple_collate_fn
from utils.loss_utils import *
from utils.dataset_utils import *
from utils.compute_utils import *

from network_pose.encoder import VGG16_bn_22333
from network_pose.decoder import PoseDecoder


class TrainerPose:
    default_cfg={
        "optimizer_type": 'adam',
        "multi_gpus": False,
        "lr_type": "exp_decay",
        "lr_cfg":1.0e-4,
        "total_step": 300000,
        "train_log_step": 20,
        "val_interval": 10000,
        "save_interval": 500,
        "worker_num": 8,
    }
    def _init_dataset(self):
        tr_transforms = transforms.Compose([
            transforms.ColorJitter(0.5,0.5,0.5,0.5),
            transforms.RandomEqualize(0.9),
            transforms.GaussianBlur(9)
        ])
        self.train_set=name2dataset[self.cfg['train_dataset_type']](self.cfg['train_dataset_cfg'], True, tr_transforms)
        self.train_set=DataLoader(self.train_set,self.cfg['batch_size'],True,num_workers=self.cfg['worker_num'],collate_fn=simple_collate_fn)
        print(f'train set len {len(self.train_set)}')
        self.val_set_list, self.val_set_names = [], []
        for val_set_cfg in self.cfg['val_set_list']:
            name, val_type, val_cfg = val_set_cfg['name'], val_set_cfg['type'], val_set_cfg['cfg']
            val_set = name2dataset[val_type](val_cfg, False)
            val_set = DataLoader(val_set,self.cfg['batch_size'],False,num_workers=self.cfg['worker_num'],collate_fn=simple_collate_fn)
            self.val_set_list.append(val_set)
            self.val_set_names.append(name)
            print(f'{name} val set len {len(val_set)}')

    def _init_network(self):
        # self.network=name2network[self.cfg['network']](self.cfg).cuda()
        def dict2namespace(config):
            namespace = argparse.Namespace()
            for key, value in config.items():
                if isinstance(value, dict):
                    new_value = dict2namespace(value)
                else:
                    new_value = value
                setattr(namespace, key, new_value)
            return namespace
        self.cfg=dict2namespace(self.cfg)
        self.encoder = VGG16_bn_22333(self.cfg.models.encoder)
        # self.encoder.cuda()
        # print(self.encoder)

        self.decoder = PoseDecoder(self.cfg.models.decoder)
        self.decoder.cuda()
        print("Decoder:")
        print(self.decoder)

        # self.val_losses = []
        # for loss_name in self.cfg['loss']:
        #     self.val_losses.append(name2loss[loss_name](self.cfg))

        # metrics
        self.val_metrics = []
        if self.cfg.val_metric in name2metrics: #val_metric: pose_errs
            self.val_metrics.append(name2metrics[self.cfg.val_metric](self.cfg)) # 一般只有pose_errs
        else:
            self.val_metrics.append(name2loss[self.cfg.val_metric](self.cfg))

        # we do not support multi gpu training
        if self.cfg.multi_gpus:
            raise NotImplementedError
            # make multi gpu network
            # self.train_network=DataParallel(MultiGPUWrapper(self.network,self.val_losses))
            # self.train_losses=[DummyLoss(self.val_losses)]
        else:
            pass
            # self.train_network=self.network
            # self.train_losses=self.val_losses

        # The optimizer
        if self.cfg.trainer.opt.type == 'adam':
            self.optimizer = optim.Adam(self.decoder.parameters(), lr=float(self.cfg.trainer.opt.lr),
                                   betas=(self.cfg.trainer.opt.beta1, self.cfg.trainer.opt.beta2),#这里两参数什么意思诶
                                   weight_decay=self.cfg.trainer.opt.weight_decay)

        if self.cfg.trainer.opt.scheduler == 'linear':
            step_size = int(getattr(self.cfg.trainer.opt, "step_epoch", 2000))

            def lambda_rule(ep):
                lr_l = 1.0 - min(1, max(0, ep - 0.5 * step_size) / float(0.45 * step_size)) * (1 - 0.01)
                return lr_l

            self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda_rule)
        else:
            assert 0, "args.schedulers should be either 'exponential' or 'linear' or 'step'"

        self.val_evaluator=ValidationEvaluator(self.cfg)
        self.lr=1e-4

    def __init__(self,cfg):
        self.cfg={**self.default_cfg,**cfg}
        self.model_name=cfg['name']
        self.model_dir=os.path.join('data/model',cfg['name'])
        if not os.path.exists(self.model_dir): os.mkdir(self.model_dir)
        self.pth_fn=os.path.join(self.model_dir,'model.pth')
        self.best_pth_fn=os.path.join(self.model_dir,'model_best.pth')

    def run(self):
        self._init_dataset()
        self._init_network()
        self._init_logger()

        start_epoch = 0
        start_time = time.time()
        #
        # if self.cfg.resume:
        #     if self.cfg.resume.pretrained is not None:
        #         start_epoch = trainer.resume(args.pretrained) + 1
        #     else:
        #         start_epoch = trainer.resume(cfg.resume.dir) + 1

        for epoch in range(start_epoch, self.cfg.trainer.epochs):
            epoch_loss = 0
            epoch_loss_t = 0
            epoch_loss_r = 0
            best_para, start_step = self._load_model()
            train_iter = iter(self.train_set)

            pbar=tqdm(total=20, bar_format='{r_bar}')
            pbar.update(start_step)
            for step in range(start_step,20):#self.cfg.total_step
                try:
                    train_data = next(train_iter)
                except StopIteration:
                    self.train_set.dataset.reset()
                    train_iter = iter(self.train_set)
                    train_data = next(train_iter)

                image_feature_map1 = self.encoder(train_data['img0'].permute(0, 3, 1, 2))
                image_feature_map2 = self.encoder(train_data['img1'].permute(0, 3, 1, 2))
                image_feature_map1 = to_cuda(image_feature_map1)
                image_feature_map2 = to_cuda(image_feature_map2)

                if not self.cfg.multi_gpus:
                    train_data = to_cuda(train_data)
                train_data['step']=step

                self.decoder.train()

                self.optimizer.zero_grad()
                self.decoder.zero_grad()

                # for loss in self.train_losses:
                #     loss_results = loss(outputs,train_data,step)
                #     for k,v in loss_results.items():
                #         log_info[k]=v

                # corr volume operation
                """这里对dimension维度归一化"""
                batch, dim, ht, wd = image_feature_map1.shape
                self.corr_volume_weight = compute_correlation_volume_weight(image_feature_map1, image_feature_map2)     # b , w*h , w*h
                self.corr_volume_weight = self.corr_volume_weight.reshape(batch, -1, 1)

                corr_volume_x0_y0_x1_y1 = compute_correlation_volume_x0_y0_x1_y1(wd=wd, ht=ht, K0=train_data['depth_K0'], K1=train_data['depth_K1'], batch_size=self.cfg.batch_size)
                self.corr_volume_x0_y0_x1_y1 = torch.from_numpy(corr_volume_x0_y0_x1_y1.astype(np.float32))
                self.corr_volume_x0_y0_x1_y1 = to_cuda(self.corr_volume_x0_y0_x1_y1)

                self.corr_volume = torch.cat([self.corr_volume_x0_y0_x1_y1.repeat(self.cfg.batch_size, 1, 1), self.corr_volume_weight], dim=2)
                # """这里有个问题，有batch怎么进行笛卡尔积构建？"""

                # loss type
                loss = 0
                # regression loss
                self.corr_volume = self.corr_volume.transpose(1, 2)
                out_r, out_t = self.decoder(self.corr_volume)
                l2_loss_r = ((out_r.view(-1) - train_data['quaternion'].view(-1)) ** 2)
                loss_r = l2_loss_r.mean()
                l2_loss_t = ((out_t.view(-1) - train_data['t_0to1'].view(-1)) ** 2)
                loss_t = l2_loss_t.mean()
                loss = (loss_r + loss_t)/2

                log_info = {}
                res = {
                    "loss": loss,
                    "loss_r": loss_r,
                    "loss_t": loss_t,
                }
                log_info.update(res)

                epoch_loss += loss
                epoch_loss_r += loss_r
                epoch_loss_t += loss_t

                loss.backward()
                self.optimizer.step()

                """val代码还要写"""
                # if (step+1)%self.cfg.val_interval==0 or (step+1)==self.cfg.total_step:
                #     torch.cuda.empty_cache()
                #     val_results, val_para = self.val_step(step)
                #
                #     if val_para>best_para:
                #         print(f'New best model {self.cfg["key_metric_name"]}: {val_para:.5f} previous {best_para:.5f}')
                #         best_para=val_para
                #         self._save_model(step+1,best_para,self.best_pth_fn)
                #     self._log_data(val_results,step+1,'val')

                pbar.set_postfix(loss=float(loss.detach().cpu().numpy()),lr=self.lr)
                pbar.update(1)

            pbar.close()

            epoch_loss_list = {}
            epoch_loss_res = {
                "epoch_loss": epoch_loss / 20,
                "epoch_loss_r": epoch_loss_r / 20,
                "epoch_loss_t": epoch_loss_t / 20,
            }
            epoch_loss_list.update(epoch_loss_res)

            # if ((step + 1) % self.cfg.train_log_step) == 0:
            self._log_data(epoch_loss_list, epoch=epoch, prefix='train_pose')
            # if (step + 1) % self.cfg.save_interval == 0:
            self._save_model(epoch=epoch)

    #
    # def val_step(self, step):
    #     val_results={}
    #     val_para = 0
    #     for vi, val_set in enumerate(self.val_set_list):
    #         val_results_cur, val_para_cur = self.val_evaluator(
    #             self.network, self.val_losses + self.val_metrics, val_set, step,
    #             self.model_name, val_set_name=self.val_set_names[vi])
    #         for k, v in val_results_cur.items():
    #             val_results[f'{self.val_set_names[vi]}-{k}'] = v
    #         # always use the final val set to select model!
    #         val_para = val_para_cur
    #     return val_results, val_para

    def _load_model(self):
        best_para,start_step=0,0
        if os.path.exists(self.pth_fn):
            checkpoint=torch.load(self.pth_fn)
            best_para = checkpoint['best_para']
            start_step = checkpoint['step']
            self.decoder.load_state_dict(checkpoint['decoder_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f'==> resuming from step {start_step} best para {best_para}')

        return best_para, start_step

    def _save_model(self, epoch=None, step=None, appendix=None):
        d = {
            'optimizer': self.optimizer.state_dict(),
            'decoder': self.decoder.state_dict(),
            'epoch': epoch,
            'step': step
        }
        # appendix这是个啥
        # if appendix is not None:
        #     d.update(appendix)
        save_name = "epoch_%s_iters_%s.pt" % (epoch, step)
        # save_name = "epoch_%s.pt" % (epoch)
        path = os.path.join(self.model_dir, "checkpoints", save_name)
        torch.save(d, path)

    def _init_logger(self):
        self.logger = Logger(self.model_dir)

    def _log_data(self,results,epoch=None,step=None,prefix='train_pose',verbose=False):
        log_results={}
        for k, v in results.items():
            if isinstance(v,float) or np.isscalar(v):
                log_results[k] = v
            elif type(v)==np.ndarray:
                log_results[k]=np.mean(v)
            else:
                log_results[k]=np.mean(v.detach().cpu().numpy())
        # self.logger.log(log_results,prefix,epoch,step,verbose)
        with open(os.path.join(self.model_dir,f'{prefix}.txt'), 'a') as f:
            msg = f'{prefix} '
            for k, v in log_results.items():
                msg += f'{k} {v:.5f} '
            msg += f'epoch={epoch}'
            f.write(msg + '\n')





