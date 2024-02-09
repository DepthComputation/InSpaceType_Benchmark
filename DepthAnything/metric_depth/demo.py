import sys
sys.path.append("../")

from pprint import pprint
import argparse
import cv2
import numpy as np
import os
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose
from tqdm import tqdm
import pandas as pd
import glob

from depth_anything.util.transform import Resize, PrepareForNet
from zoedepth.utils.config import get_config
from zoedepth.models.builder import build_model

import matplotlib as mpl
import matplotlib.cm as cm

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
    parser = argparse.ArgumentParser()
    parser.add_argument('--img-path', type=str)
    parser.add_argument('--outdir', type=str, default='./vis_depth')
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl'])
    
    args = parser.parse_args()
    
    margin_width = 50
    caption_height = 60
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

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
    
    # Different from foundation model, the pretrained metric model is initlized via a config file
    # See https://github.com/LiheYoung/Depth-Anything/blob/main/metric_depth/README.md        
    pretrained_resource = 'local::./checkpoints/depth_anything_metric_depth_indoor.pt'
    overwrite = {"pretrained_resource": pretrained_resource}
    # use the config for finetuning on NYUv2 dataset
    config = get_config('zoedepth', "eval", 'nyu', **overwrite)
    pprint(config)
    depth_anything = build_model(config).cuda().eval()
     
    total_params = sum(param.numel() for param in depth_anything.parameters())
    print('Total parameters: {:.2f}M'.format(total_params / 1e6))

    transform = Compose([
        Resize(
            width=518,
            height=392,
            resize_target=False,
            keep_aspect_ratio=True,
            ensure_multiple_of=14,
            resize_method='lower_bound',
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        # finetued metric models did not use normalization
        PrepareForNet(),
    ])
    
    if os.path.isfile(args.img_path):
        if args.img_path.endswith('txt'):
            with open(args.img_path, 'r') as f:
                filenames = f.read().splitlines()
        else:
            filenames = [args.img_path]
    else:
        # fetch all left images in the InSpaceType folder
        filenames = glob.glob(args.img_path+'/*_L.jpg')
        filenames.sort()
    
    os.makedirs(args.outdir, exist_ok=True)

    depth_metric_names = [
        "de/abs_mn", "de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]
    N = 1
    print("start processing")
    
    for ind, filename in tqdm(enumerate(filenames)):
        raw_image = cv2.imread(filename)
        image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0
        
        h, w = image.shape[:2]
        
        image = transform({'image': image})['image']
        image = torch.from_numpy(image).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            depth = depth_anything(image)['metric_depth']

        depth = F.interpolate(depth, (h, w), mode='bilinear', align_corners=False)[0, 0]
        depth = depth.cpu()
        name = filename.rsplit('/', 1)[-1][:-4]
        # plot 0.001 to 5.0 meters
        write_turbo_depth_metric(args.outdir+f'/{name}.png', depth, vmin=0.001, vmax=5.0)

        depth_gt = cv2.imread(f'{filename.replace("_L.jpg",".pfm")}',-1)
        depth_gt = cv2.resize(depth_gt, (2208, 1242), interpolation=cv2.INTER_NEAREST)
        depth_gt = torch.Tensor(depth_gt)
        mask = torch.logical_and(depth_gt > 0.01, depth_gt<=10.0)

        depth_gt = depth_gt[mask]
        prediction_p = depth[mask]

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
    f = open(f'evaluation-anything-all.txt','w')
    metric_set = global_metr[i]
    for num in metric_set:
        f.write(str(num))
    f.close()

for i in range(4):
    f = open(f'evaluation-anything-H1_{i+1}.txt','w')
    metric_set = H0_metr[i]
    for num in metric_set:
        f.write(str(num))
    f.close()

for i in range(12):
    f = open(f'evaluation-anything-H2_{i+1}.txt','w')
    metric_set = H1_metr[i]
    for num in metric_set:
        f.write(str(num))
    f.close()

for i in range(27):
    f = open(f'evaluation-anything-H3_{i+1}.txt','w')
    metric_set = H2_metr[i]
    for num in metric_set:
        f.write(str(num))
    f.close()

print("Finished. All performance reports (H0-H2) are saved under the current directory.")