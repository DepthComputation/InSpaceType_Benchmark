# ------------------------------------------------------------------------------
# The code is from GLPDepth (https://github.com/vinvino02/GLPDepth).
# For non-commercial purpose only (research, evaluation etc).
# Shift window testing and flip testing is modified by Zigang Geng (zigang@mail.ustc.edu.cn).
# -----------------------------------------------------------------------------

import os
import cv2
import numpy as np
from collections import OrderedDict

import torch
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn

import utils.logging as logging
import utils.metrics as metrics
from models.model import GLPDepth
from dataset.base_dataset import get_dataset
from configs.test_options import TestOptions
import matplotlib as mpl
import matplotlib.cm as cm
import pandas as pd
from tqdm import tqdm
metric_name = ['d1', 'd2', 'd3', 'abs_rel', 'sq_rel', 'rmse', 'rmse_log',
               'log10', 'silog']

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

def main():
    # experiments setting
    opt = TestOptions()
    args = opt.initialize().parse_args()
    print(args)

    if args.gpu_or_cpu == 'gpu':
        device = torch.device('cuda')
        cudnn.benchmark = True
    else:
        device = torch.device('cpu')

    # if args.save_eval_pngs or args.save_visualize:
    #     result_path = os.path.join(args.result_dir, args.exp_name)
    #     logging.check_and_make_dirs(result_path)
    #     print("Saving result images in to %s" % result_path)
    
    # if args.do_evaluate:
    #     result_metrics = {}
    #     for metric in metric_name:
    #         result_metrics[metric] = 0.0

    print("\n1. Define Model")
    model = GLPDepth(args=args).to(device)
    
    model_weight = torch.load(args.ckpt_dir)
    if 'module' in next(iter(model_weight.items()))[0]:
        model_weight = OrderedDict((k[7:], v) for k, v in model_weight.items())
    model.load_state_dict(model_weight)
    model.eval()

    print("\n2. Define Dataloader")
    if args.dataset == 'imagepath': # not for do_evaluate in case of imagepath
        dataset_kwargs = {'dataset_name': 'ImagePath', 'data_path': args.data_path}
    else:
        dataset_kwargs = {'data_path': args.data_path, 'dataset_name': args.dataset,
                          'is_train': False}

    test_dataset = get_dataset(**dataset_kwargs)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                             pin_memory=True)

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

    print("\n3. Inference & Evaluate")
    ind = 0 
    save_name = 'output_monodepth'
    for batch_idx, batch in enumerate(tqdm(test_loader)):
        input_RGB = batch['image'].to(device)
        input_RGB = torch.nn.functional.interpolate(input_RGB, (416,736),mode='bilinear')
        filename = batch['filename'][0]

        with torch.no_grad():
            if args.shift_window_test:
                bs, _, h, w = input_RGB.shape
                assert w > h and bs == 1
                interval_all = w - h
                interval = interval_all // (args.shift_size-1)
                sliding_images = []
                sliding_masks = torch.zeros((bs, 1, h, w), device=input_RGB.device) 
                for i in range(args.shift_size):
                    sliding_images.append(input_RGB[..., :, i*interval:i*interval+h])
                    sliding_masks[..., :, i*interval:i*interval+h] += 1
                input_RGB = torch.cat(sliding_images, dim=0)
            if args.flip_test:
                input_RGB = torch.cat((input_RGB, torch.flip(input_RGB, [3])), dim=0)
            pred = model(input_RGB)
        pred_d = pred['pred_d']
        if args.flip_test:
            batch_s = pred_d.shape[0]//2
            pred_d = (pred_d[:batch_s] + torch.flip(pred_d[batch_s:], [3]))/2.0
        if args.shift_window_test:
            pred_s = torch.zeros((bs, 1, h, w), device=pred_d.device)
            for i in range(args.shift_size):
                pred_s[..., :, i*interval:i*interval+h] += pred_d[i:i+1]
            pred_d = pred_s/sliding_masks

        pred_d = torch.nn.functional.interpolate(pred_d, (1242,2208),mode='bilinear')

        depth_gt = batch['depth']
        pred_d, depth_gt = pred_d.squeeze().cpu(), depth_gt.squeeze()

        mask = torch.logical_and(depth_gt > 0.01, depth_gt<=10.0)
        mask[:20,:] = False
        depth_gt = depth_gt[mask]
        prediction_p = pred_d[mask]
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
        f = open(f'evaluation-mim-all.txt','w')
        metric_set = global_metr[i]
        for num in metric_set:
            f.write(str(num))
        f.close()

    for i in range(4):
        f = open(f'evaluation-mim-H0_{i+1}.txt','w')
        metric_set = H0_metr[i]
        for num in metric_set:
            f.write(str(num))
        f.close()

    for i in range(12):
        f = open(f'evaluation-mim-H1_{i+1}.txt','w')
        metric_set = H1_metr[i]
        for num in metric_set:
            f.write(str(num))
        f.close()
    
    for i in range(27):
        f = open(f'evaluation-mim-H2_{i+1}.txt','w')
        metric_set = H2_metr[i]
        for num in metric_set:
            f.write(str(num))
        f.close()

    print("Done")


if __name__ == "__main__":
    main()
