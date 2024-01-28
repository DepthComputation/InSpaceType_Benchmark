import os

from dataloader import DepthDataLoader
import argparse
import torch
import numpy as np
import torch.nn as nn
import models.models

import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib as mpl
import matplotlib.cm as cm
import pandas as pd
import cv2

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


def predict_tta(model, image, device, args, inv_depth):
    input_image_size = [args.input_height, args.input_width]
    interpolate_bilinear_input_size = nn.Upsample(size=input_image_size, mode='bilinear', align_corners=True)
    image_input = interpolate_bilinear_input_size(image)

    image_input = image_input.to(device)
    pred = model(image_input)

    relu = nn.ReLU()
    if inv_depth:
        pred = (1 / pred) - 1
    pred = relu(pred - 0.0000000001) + 0.0000000001

    ####################################################
    pred = np.clip(pred.cpu().numpy(), args.min_depth, args.max_depth)

    image_input = torch.Tensor(np.array(image_input.cpu().numpy())[..., ::-1].copy()).to(device)

    pred_lr = model(image_input)
    relu = nn.ReLU()
    if inv_depth:
        pred_lr = (1 / pred_lr) - 1
    pred_lr = relu(pred_lr - 0.0000000001) + 0.0000000001
    pred_lr = np.clip(pred_lr.cpu().numpy()[..., ::-1], args.min_depth, args.max_depth)

    final = 0.5 * (pred + pred_lr)
    final = nn.functional.interpolate(torch.Tensor(final), image.shape[-2:], mode='bilinear', align_corners=True)

    return torch.Tensor(final)


def eval(model, test_loader, args, gpus=None, inv_depth=False):
    if gpus is None:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    elif gpus == 'cpu':
        device = torch.device('cpu')
    else:
        device = gpus[0]

    meta = pd.read_csv('../InSpaceType_meta.csv')

    global_metr = []
    N = 1
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

    with torch.no_grad():
        model.eval()

        sequential = test_loader
        for ind, batch in enumerate(tqdm(sequential)):
            image = batch['image'].to(device)
            gt = batch['depth'].to(device)

            final = predict_tta(model, image, device, args, inv_depth)
            final = final.squeeze().cpu()
            
            final[torch.isinf(final)] = args.max_depth
            final[torch.isnan(final)] = args.min_depth

            gt = gt.squeeze().cpu()
            valid_mask = torch.logical_and(gt > args.min_depth, gt < args.max_depth)

            depth_gt = gt[valid_mask]
            prediction_p = final[valid_mask]
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


    for i in range(1):
        f = open(f'evaluation-decompose-all.txt','w')
        metric_set = global_metr[i]
        for num in metric_set:
            f.write(str(num))
        f.close()

    for i in range(4):
        f = open(f'evaluation-decompose-H0_{i+1}.txt','w')
        metric_set = H0_metr[i]
        for num in metric_set:
            f.write(str(num))
        f.close()

    for i in range(12):
        f = open(f'evaluation-decompose-H1_{i+1}.txt','w')
        metric_set = H1_metr[i]
        for num in metric_set:
            f.write(str(num))
        f.close()
    
    for i in range(27):
        f = open(f'evaluation-decompose-H2_{i+1}.txt','w')
        metric_set = H2_metr[i]
        for num in metric_set:
            f.write(str(num))
        f.close()

def parse_args():
    parser = argparse.ArgumentParser(description='Model evaluator', fromfile_prefix_chars='@',
                                     conflict_handler='resolve')
    parser.add_argument('--gpu', default=None, type=int, help='Which gpu to use')

    parser.add_argument("--dataset", default='nyu', type=str, help="Dataset to train on")
    parser.add_argument("--data_path", default='nyu_depth_v2/official_splits/test/', type=str,
                        help="path to dataset")
    parser.add_argument("--gt_path", default='nyu_depth_v2/official_splits/test/', type=str,
                        help="path to dataset gt")

    parser.add_argument('--filenames_file',
                        default="nyu_depth_v2/official_splits/test/nyudepthv2_test_files_with_gt.txt",
                        type=str, help='path to the filenames text file')

    parser.add_argument('--input_height', type=int, help='input height', default=384)
    parser.add_argument('--input_width', type=int, help='input width', default=512)
    parser.add_argument('--max_depth', type=float, help='maximum depth in estimation', default=10)
    parser.add_argument('--min_depth', type=float, help='minimum depth in estimation', default=1e-3)

    parser.add_argument('--ckpt',
                        default='51k.pth',
                        type=str,
                        help="checkpoint file to use for prediction")

    parser.add_argument('--min_depth_eval', type=float, help='minimum depth for evaluation', default=1e-3)
    parser.add_argument('--max_depth_eval', type=float, help='maximum depth for evaluation', default=10)
    parser.add_argument('--eigen_crop', help='if set, crops according to Eigen NIPS14', action='store_true')

    args = parser.parse_args()

    return args


# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = str(0)

def test(args):
    args.gpu = int(args.gpu) if args.gpu is not None else 0
    args.distributed = False
    device = torch.device('cuda:{}'.format(args.gpu))

    test = DepthDataLoader(args, 'online_eval').data

    if 'HRWSI' in args.ckpt:
        model_name = 'depth_decomp_effB5'
        inv_depth = True
        if '795' in args.ckpt:
            MDR = False
        else:
            MDR = True
    else:
        model_name = 'depth_effB5'
        inv_depth = False
        MDR = False

    model = models.models.create_model(model_name=model_name, MDR=MDR).cuda()

    model.load_state_dict(torch.load(args.ckpt, map_location='cpu')['model'], strict=False)
    model = model.eval()

    eval(model, test, args, gpus=[device], inv_depth=inv_depth)


if __name__ == '__main__':    
    args = parse_args()
    test(args)