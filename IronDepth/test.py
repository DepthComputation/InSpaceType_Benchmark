import argparse
import os
import sys
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

import utils.utils as utils

import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
import matplotlib as mpl
import matplotlib.cm as cm
import pandas as pd

class AverageMeter(object):
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} :{avg' + self.fmt + '}\n'
        return fmtstr.format(**self.__dict__)

def write_turbo_depth_metric(path, toplot, vmin=0.001, vmax=5.0):
    v_min = vmin
    v_max = vmax
    normalizer = mpl.colors.Normalize(vmin=v_min, vmax=v_max)
    mapper = cm.ScalarMappable(norm=normalizer, cmap='turbo')
    colormapped_im = (mapper.to_rgba(toplot)[:,:,:3]*255).astype(np.uint8)
    cv2.imwrite(path, colormapped_im[:,:,[2,1,0]])

def compute_depth_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths"""
    thresh = torch.max((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).float().mean()
    a2 = (thresh < 1.25 ** 2).float().mean()
    a3 = (thresh < 1.25 ** 3).float().mean()

    abs_mn = torch.abs(gt-pred).mean()

    rmse = (gt - pred) ** 2
    rmse = torch.sqrt(rmse.mean())

    rmse_log = (torch.log(gt) - torch.log(pred)) ** 2
    rmse_log = torch.sqrt(rmse_log.mean())

    abs_rel = torch.mean(torch.abs(gt - pred) / gt)

    sq_rel = torch.mean((gt - pred) ** 2 / gt)

    return abs_mn, abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3

if __name__ == '__main__':
    # Arguments
    parser = argparse.ArgumentParser()

    # model architecture
    parser.add_argument('--train_data', type=str, default='scannet', help='{nyuv2, scannet}')
    parser.add_argument("--test_data", type=str, default='scannet', help="{nyuv2, scannet, custom}")

    parser.add_argument('--NNET_architecture', type=str, default=None)
    parser.add_argument('--NNET_ckpt', type=str, default=None)
    parser.add_argument('--IronDepth_ckpt', type=str, default=None)

    parser.add_argument('--train_iter', type=int, default=3)
    parser.add_argument('--test_iter', type=int, default=20)
    args = parser.parse_args()

    if args.train_data == 'scannet':
        args.NNET_architecture = 'BN'
        args.NNET_ckpt = './checkpoints/normal_scannet.pt'
        args.IronDepth_ckpt = './checkpoints/irondepth_scannet.pt'
    elif args.train_data == 'nyuv2':
        args.NNET_architecture = 'GN'
        args.NNET_ckpt = './checkpoints/normal_nyuv2.pt'
        args.IronDepth_ckpt = './checkpoints/irondepth_nyuv2.pt'

    meta = pd.read_csv('../InSpaceType_meta.csv')

    global_metr = []
    for _ in range(1):
        local_metr = []
        local_metr.append(AverageMeter('abs_mean'))
        local_metr.append(AverageMeter('abs_rel'))
        local_metr.append(AverageMeter('sq_rel'))
        local_metr.append(AverageMeter('rms'))
        local_metr.append(AverageMeter('log_rms'))
        local_metr.append(AverageMeter('a1'))
        local_metr.append(AverageMeter('a2'))
        local_metr.append(AverageMeter('a3'))
        global_metr.append(local_metr)

    H0_metr = []
    for _ in range(4):
        local_metr = []
        local_metr.append(AverageMeter('abs_mean'))
        local_metr.append(AverageMeter('abs_rel'))
        local_metr.append(AverageMeter('sq_rel'))
        local_metr.append(AverageMeter('rms'))
        local_metr.append(AverageMeter('log_rms'))
        local_metr.append(AverageMeter('a1'))
        local_metr.append(AverageMeter('a2'))
        local_metr.append(AverageMeter('a3'))
        H0_metr.append(local_metr)

    H1_metr = []
    for _ in range(12):
        local_metr = []
        local_metr.append(AverageMeter('abs_mean'))
        local_metr.append(AverageMeter('abs_rel'))
        local_metr.append(AverageMeter('sq_rel'))
        local_metr.append(AverageMeter('rms'))
        local_metr.append(AverageMeter('log_rms'))
        local_metr.append(AverageMeter('a1'))
        local_metr.append(AverageMeter('a2'))
        local_metr.append(AverageMeter('a3'))
        H1_metr.append(local_metr)

    H2_metr = []
    for _ in range(27):
        local_metr = []
        local_metr.append(AverageMeter('abs_mean'))
        local_metr.append(AverageMeter('abs_rel'))
        local_metr.append(AverageMeter('sq_rel'))
        local_metr.append(AverageMeter('rms'))
        local_metr.append(AverageMeter('log_rms'))
        local_metr.append(AverageMeter('a1'))
        local_metr.append(AverageMeter('a2'))
        local_metr.append(AverageMeter('a3'))
        H2_metr.append(local_metr)


    depth_metric_names = [
        "de/abs_mn", "de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]
    N = 1

    device = torch.device('cuda:0')

    # define N_NET (surface normal estimation network)
    from models_normal.NNET import NNET
    n_net = NNET(args).to(device)
    print('loading N-Net weights from %s' % args.NNET_ckpt)
    n_net = utils.load_checkpoint(args.NNET_ckpt, n_net)
    n_net.eval()

    # define IronDepth
    from models.IronDepth import IronDepth
    model = IronDepth(args).to(device)
    print('loading IronDepth weights from %s' % args.IronDepth_ckpt)
    model = utils.load_checkpoint(args.IronDepth_ckpt, model)
    model.eval()

    # define dataloader
    from data.dataloader_custom_rev import CustomLoader
    test_loader = CustomLoader('../InSpaceType').data

    # output dir
    # output_dir = './examples/output/%s/' % args.test_data
    # os.makedirs(output_dir, exist_ok=True)

    save_name = 'output_monodepth'

    ind = 0 
    with torch.no_grad():
        for data_dict in tqdm(test_loader):
            img_name = data_dict['img_name'][0]

            img = data_dict['img'].to(device)
            pos = data_dict['pos'].to(device)
            # surface normal prediction
            norm_out = n_net(img)
            pred_norm = norm_out[:, :3, :, :]
            pred_kappa = norm_out[:, 3:, :, :]

            input_dict = {
                'img': img,
                'pred_norm': pred_norm,
                'pred_kappa': pred_kappa,
                'pos': pos,
            }

            # IronDepth forward pass
            pred_list = model(input_dict, 'test')

            pred_dmap = pred_list[-1].detach().cpu().float()
            pred_dmap = torch.nn.functional.interpolate(pred_dmap, (1242,2208),mode='bilinear')
            pred_dmap = pred_dmap[0,0, ...]
            
            pred_depth = pred_dmap

            depth_gt = data_dict['depth_gt'].squeeze()
            mask = torch.logical_and(depth_gt > 0.01, depth_gt<=10.0)

            depth_gt = depth_gt[mask]
            prediction_p = pred_depth[mask]
            prediction_p *= torch.median(depth_gt) / torch.median(prediction_p)
            depth_errors = compute_depth_errors(depth_gt, prediction_p)

            metric_set_global = global_metr[0]
            for i, var in enumerate(metric_set_global):
                var.update(np.array(depth_errors[i].cpu()), N)

            H0_cat = meta['H0'][ind + 1]-1
            H0_metr_set = H0_metr[H0_cat]
            for i, var in enumerate(H0_metr_set):
                var.update(np.array(depth_errors[i].cpu()), N)

            H1_cat = meta['H1'][ind + 1]-1
            H1_metr_set = H1_metr[H1_cat]
            for i, var in enumerate(H1_metr_set):
                var.update(np.array(depth_errors[i].cpu()), N)

            H2_cat = meta['H2'][ind + 1]-1
            H2_metr_set = H2_metr[H2_cat]
            for i, var in enumerate(H2_metr_set):
                var.update(np.array(depth_errors[i].cpu()), N)

            ind += 1

    for i in range(1):
        f = open(f'evaluation-irondepth-all.txt','w')
        metric_set = global_metr[i]
        for num in metric_set:
            f.write(str(num))
        f.close()

    for i in range(4):
        f = open(f'evaluation-irondepth-H0_{i+1}.txt','w')
        metric_set = H0_metr[i]
        for num in metric_set:
            f.write(str(num))
        f.close()

    for i in range(12):
        f = open(f'evaluation-irondepth-H1_{i+1}.txt','w')
        metric_set = H1_metr[i]
        for num in metric_set:
            f.write(str(num))
        f.close()
    
    for i in range(27):
        f = open(f'evaluation-irondepth-H2_{i+1}.txt','w')
        metric_set = H2_metr[i]
        for num in metric_set:
            f.write(str(num))
        f.close()

