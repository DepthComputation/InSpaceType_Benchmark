import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from utils import RunningAverageDict
from scipy import stats


def compute_errors(gt, pred):
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    err = np.log(pred) - np.log(gt)
    silog = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2) * 100

    log_10 = (np.abs(np.log10(gt) - np.log10(pred))).mean()
    k_tau, p_value = stats.kendalltau(pred, gt)

    return dict(a1=a1, a2=a2, a3=a3, abs_rel=abs_rel, rmse=rmse, log_10=log_10, rmse_log=rmse_log,
                silog=silog, sq_rel=sq_rel, k_tau=k_tau)


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

    metrics = RunningAverageDict()
    total_invalid = 0
    with torch.no_grad():
        model.eval()

        sequential = test_loader
        for batch in tqdm(sequential):
            image = batch['image'].to(device)
            gt = batch['depth'].to(device)

            final = predict_tta(model, image, device, args, inv_depth)
            final = final.squeeze().cpu().numpy()
            
            final[np.isinf(final)] = args.max_depth
            final[np.isnan(final)] = args.min_depth

            gt = gt.squeeze().cpu().numpy()
            valid_mask = np.logical_and(gt > args.min_depth, gt < args.max_depth)

            # NYU crop
            eval_mask = np.zeros(valid_mask.shape)
            eval_mask[45:471, 41:601] = 1  # NYU crop
            valid_mask = np.logical_and(valid_mask, eval_mask)

            metrics.update(compute_errors(gt[valid_mask], final[valid_mask]))

    print(f"Total invalid: {total_invalid}")
    metrics_ = {k: round(v, 3) for k, v in metrics.get_value().items()}
    print(f"Metrics: {metrics_}")

