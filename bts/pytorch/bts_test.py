# Copyright (C) 2019 Jin Han Lee
#
# This file is a part of BTS.
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>

from __future__ import absolute_import, division, print_function

import os
import argparse
import time
import numpy as np
import cv2
import sys

import torch
import torch.nn as nn
from torch.autograd import Variable
from bts_dataloader import *

import errno
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib as mpl
import matplotlib.cm as cm
import pandas as pd
from bts_dataloader import *


def convert_arg_line_to_args(arg_line):
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield arg

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

parser = argparse.ArgumentParser(description='BTS PyTorch implementation.', fromfile_prefix_chars='@')
parser.convert_arg_line_to_args = convert_arg_line_to_args

parser.add_argument('--model_name', type=str, help='model name', default='bts_nyu_v2')
parser.add_argument('--encoder', type=str, help='type of encoder, vgg or desenet121_bts or densenet161_bts',
                    default='densenet161_bts')
parser.add_argument('--data_path', type=str, help='path to the data')
parser.add_argument('--filenames_file', type=str, help='path to the filenames text file', required=True)
parser.add_argument('--input_height', type=int, help='input height', default=480)
parser.add_argument('--input_width', type=int, help='input width', default=640)
parser.add_argument('--max_depth', type=float, help='maximum depth in estimation', default=80)
parser.add_argument('--checkpoint_path', type=str, help='path to a specific checkpoint to load', default='')
parser.add_argument('--dataset', type=str, help='dataset to train on, make3d or nyudepthv2', default='nyu')
parser.add_argument('--do_kb_crop', help='if set, crop input images as kitti benchmark images', action='store_true')
parser.add_argument('--save_lpg', help='if set, save outputs from lpg layers', action='store_true')
parser.add_argument('--bts_size', type=int,   help='initial num_filters in bts', default=512)

if sys.argv.__len__() == 2:
    arg_filename_with_prefix = '@' + sys.argv[1]
    args = parser.parse_args([arg_filename_with_prefix])
else:
    args = parser.parse_args()

model_dir = os.path.dirname(args.checkpoint_path)
sys.path.append(model_dir)

for key, val in vars(__import__(args.model_name)).items():
    if key.startswith('__') and key.endswith('__'):
        continue
    vars()[key] = val


def get_num_lines(file_path):
    f = open(file_path, 'r')
    lines = f.readlines()
    f.close()
    return len(lines)


def test(params):
    """Test function."""
    args.mode = 'test'
    dataloader = BtsDataLoader(args, 'test')
    
    model = BtsModel(params=args)
    model = torch.nn.DataParallel(model)
    
    checkpoint = torch.load(args.checkpoint_path)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    model.cuda()

    num_params = sum([np.prod(p.size()) for p in model.parameters()])
    print("Total number of parameters: {}".format(num_params))

    num_test_samples = get_num_lines(args.filenames_file)

    with open(args.filenames_file) as f:
        lines = f.readlines()

    print('now testing {} files with {}'.format(num_test_samples, args.checkpoint_path))

    meta = pd.read_csv('../../InSpaceType_meta.csv')

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

    N = 1

    pred_depths = []


    start_time = time.time()
    with torch.no_grad():
        for _, sample in enumerate(tqdm(dataloader.data)):
            image = Variable(sample['image'].cuda())
            #image = torch.nn.functional.interpolate(image, (480,640),mode='bilinear')
            image = torch.nn.functional.interpolate(image, (416,736),mode='bilinear')
            focal = Variable(sample['focal'].cuda())
            # Predict
            lpg8x8, lpg4x4, lpg2x2, reduc1x1, depth_est = model(image, focal)
            depth_est = torch.nn.functional.interpolate(depth_est, (1242,2208),mode='bilinear')
            pred_depths.append(depth_est.cpu().squeeze())

    elapsed_time = time.time() - start_time
    print('Elapesed time: %s' % str(elapsed_time))
    print('Done.')
    
    save_name = 'result_' + args.model_name
    
    save_name = 'output_monodepth'
    
    if not os.path.exists(save_name):
        try:
            os.mkdir(save_name)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
    
    ind = 0 
    for s in tqdm(range(num_test_samples)):
        scene_name = lines[s].rsplit('/',1)[-1][:-5]
        
        gt_path = lines[s].replace('_L.jpg\n','.pfm')
        gt = cv2.imread(gt_path, -1)
        depth_gt = torch.Tensor(gt)
        mask = torch.logical_and(depth_gt > 0.01, depth_gt<=10.0)
        
        pred_depth = pred_depths[s]

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
        f = open(f'evaluation-bts-all.txt','w')
        metric_set = global_metr[i]
        for num in metric_set:
            f.write(str(num))
        f.close()

    for i in range(4):
        f = open(f'evaluation-bts-H0_{i+1}.txt','w')
        metric_set = H0_metr[i]
        for num in metric_set:
            f.write(str(num))
        f.close()

    for i in range(12):
        f = open(f'evaluation-bts-H1_{i+1}.txt','w')
        metric_set = H1_metr[i]
        for num in metric_set:
            f.write(str(num))
        f.close()
    
    for i in range(27):
        f = open(f'evaluation-bts-H2_{i+1}.txt','w')
        metric_set = H2_metr[i]
        for num in metric_set:
            f.write(str(num))
        f.close()




if __name__ == '__main__':
    test(args)
