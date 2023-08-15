import torch
import cv2
import matplotlib as mpl
import matplotlib.cm as cm
import numpy as np
#from util.misc import visualize_attention
import glob
import os
import pandas as pd
import argparse
from PIL import Image
from zoedepth.utils.misc import colorize
# Zoe_N
model_zoe_n = torch.hub.load(".", "ZoeD_N", source="local", pretrained=True)


parser = argparse.ArgumentParser()

parser.add_argument(
    "-i", "--input_path", default="input", help="folder with input images"
)

parser.add_argument(
    "-o",
    "--output_path",
    default="output_monodepth",
    help="folder for output images",
)

args = parser.parse_args()

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

# Zoe_N
model_zoe_n = torch.hub.load(".", "ZoeD_N", source="local", pretrained=True)

##### sample prediction
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
zoe = model_zoe_n.to(DEVICE)

img_names = sorted(glob.glob(os.path.join(args.input_path, "*_L.jpg")))
num_images = len(img_names)

# create output folder
os.makedirs(args.output_path, exist_ok=True)

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


N=1
for ind, img_name in enumerate(img_names):
    if os.path.isdir(img_name):
        continue

    print("  processing {} ({}/{})".format(img_name, ind + 1, num_images))
    # input

    image = Image.open(img_name).convert("RGB")
    prediction = zoe.infer_pil(image, output_type="tensor")


    filename = os.path.join(
        args.output_path, os.path.splitext(os.path.basename(img_name))[0]
    )
    depth_gt = cv2.imread(f'{img_name.replace("_L.jpg",".pfm")}',-1)
    depth_gt = torch.Tensor(depth_gt)
    mask = torch.logical_and(depth_gt > 0.01, depth_gt<=10.0)

    depth_gt = depth_gt[mask]
    prediction_p = prediction[mask]

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
    f = open(f'evaluation-zoe-global_{i}.txt','w')
    metric_set = global_metr[i]
    for num in metric_set:
        f.write(str(num))
    f.close()

for i in range(4):
    f = open(f'evaluation-zoe-H0_{i}.txt','w')
    metric_set = H0_metr[i]
    for num in metric_set:
        f.write(str(num))
    f.close()

for i in range(12):
    f = open(f'evaluation-zoe-H1_{i}.txt','w')
    metric_set = H1_metr[i]
    for num in metric_set:
        f.write(str(num))
    f.close()

for i in range(27):
    f = open(f'evaluation-zoe-H2_{i}.txt','w')
    metric_set = H2_metr[i]
    for num in metric_set:
        f.write(str(num))
    f.close()

print("finished")