# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import cv2
import matplotlib as mpl
import matplotlib.cm as cm
import numpy as np
import os
import torch
from networks.depth_decoder import DepthDecoder
from networks.resnet_encoder import ResnetEncoder
from utils import output_to_depth, AverageMeter, write_turbo_depth_metric
from layers import compute_depth_errors
import pandas as pd
torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":

    dir_prefix = "./"
    file_list = open("split_files.txt", "r")
    files = file_list.readlines()
    output_path = dir_prefix + "results"

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

    save_name = 'output_monodepth'

    with torch.no_grad():

        print("Loading the pretrained network")
        encoder = ResnetEncoder(152, False)
        loaded_dict_enc = torch.load(
            dir_prefix + "ckpts-finetuned/encoder.pth",
            map_location=device,
        )

        filtered_dict_enc = {
            k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()
        }
        encoder.load_state_dict(filtered_dict_enc)
        encoder.to(device)
        encoder.eval()

        depth_decoder = DepthDecoder(num_ch_enc=encoder.num_ch_enc, scales=range(4))

        loaded_dict = torch.load(
            dir_prefix + "ckpts-finetuned/depth.pth",
            map_location=device,
        )
        depth_decoder.load_state_dict(loaded_dict)

        depth_decoder.to(device)
        depth_decoder.eval()

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        for idx, file in enumerate(files):
            raw_img = np.transpose(
                cv2.imread(file[:-1], -1)[:, :, :3], (2, 0, 1)
            )
            input_image = torch.from_numpy(raw_img).float().to(device)
            input_image = (input_image / 255.0).unsqueeze(0)

            input_image = torch.nn.functional.interpolate(
                input_image, (256, 256), mode="bilinear", align_corners=False
            )
            features = encoder(input_image)
            outputs = depth_decoder(features)

            out = outputs[("out", 0)]
            out_resized = torch.nn.functional.interpolate(
                out, (1242, 2208), mode="bilinear", align_corners=False
            )
            depth = output_to_depth(out_resized, 0.1, 10)
            metric_depth = depth.cpu().squeeze()#.numpy()

            gt_path = file.replace('_L.jpg\n','.pfm')
            gt = cv2.imread(gt_path, -1)
            depth_gt = torch.Tensor(gt)
            mask = torch.logical_and(depth_gt > 0.01, depth_gt<=10.0)

            depth_gt = depth_gt[mask]
            prediction_p = metric_depth[mask]
            prediction_p *= torch.median(depth_gt) / torch.median(prediction_p)
            depth_errors = compute_depth_errors(depth_gt, prediction_p)

            metric_set_global = global_metr[0]
            for i, var in enumerate(metric_set_global):
                var.update(np.array(depth_errors[i].cpu()), N)

            H0_cat = meta['H0'][idx + 1]-1
            H0_metr_set = H0_metr[H0_cat]
            for i, var in enumerate(H0_metr_set):
                var.update(np.array(depth_errors[i].cpu()), N)

            H1_cat = meta['H1'][idx + 1]-1
            H1_metr_set = H1_metr[H1_cat]
            for i, var in enumerate(H1_metr_set):
                var.update(np.array(depth_errors[i].cpu()), N)

            H2_cat = meta['H2'][idx + 1]-1
            H2_metr_set = H2_metr[H2_cat]
            for i, var in enumerate(H2_metr_set):
                var.update(np.array(depth_errors[i].cpu()), N)


        for i in range(1):
            f = open(f'evaluation-distdepthNYU-all.txt','w')
            metric_set = global_metr[i]
            for num in metric_set:
                f.write(str(num))
            f.close()

        for i in range(4):
            f = open(f'evaluation-distdepthNYU-H0_{i+1}.txt','w')
            metric_set = H0_metr[i]
            for num in metric_set:
                f.write(str(num))
            f.close()

        for i in range(12):
            f = open(f'evaluation-distdepthNYU-H1_{i+1}.txt','w')
            metric_set = H1_metr[i]
            for num in metric_set:
                f.write(str(num))
            f.close()

        for i in range(27):
            f = open(f'evaluation-distdepthNYU-H2_{i+1}.txt','w')
            metric_set = H2_metr[i]
            for num in metric_set:
                f.write(str(num))
            f.close()