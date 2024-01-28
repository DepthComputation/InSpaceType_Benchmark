import os

from dataloader import DepthDataLoader
import argparse
import torch

from eval import eval
import models.models


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