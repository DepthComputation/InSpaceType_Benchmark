# Benchamrk on paper PixelFormer: Attention Attention Everywhere: Monocular Depth Prediction with Skip Attention

1. Download InSpaceType eval set. Install [torch and torchivsion](https://pytorch.org/get-started/previous-versions/) and packages: matplotlib, tqdm, tensorboardX, timm, mmcv, opencv-python

2. Download pretrained model 'nyu.pt' from [Official Link](https://drive.google.com/drive/folders/1Feo67jEbccqa-HojTHG7ljTXOW2yuX-X?usp=share_link) and put it here

3.

  ```
  python pixelformer/test.py --data_path ./ --dataset nyu --filenames_file data_splits/split_files.txt --checkpoint_path nyu.pth --max_depth 10
  ```

  The command generates report files for hierarchy (H0-H2). *-all means overall H0-H2 means level of hierarchy. H1_xx means scene space type number. See [space_type_def.yml](https://github.com/DepthComputation/InSpaceType_Benchmark/blob/main/space_type_def.yml) for space type number definition.