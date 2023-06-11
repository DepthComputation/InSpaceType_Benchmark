# Benchamrk on paper  NeW CRFs: Neural Window Fully-connected CRFs for Monocular Depth Estimation

1. Download InSpaceType eval set. Install [torch and torchivsion](https://pytorch.org/get-started/previous-versions/) and packages: matplotlib, tqdm, tensorboardX, timm, mmcv, opencv-python

2. Download pretrained model 'model_nyu.ckpt' from [Official Link](https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/newcrfs/models/model_nyu.ckpt) and put it here

3.

  ```  
  python newcrfs/test.py --data_path ./ --dataset nyu --filenames_file data_splits/split_files.txt --checkpoint_path model_nyu.ckpt --max_depth 10
  ```
