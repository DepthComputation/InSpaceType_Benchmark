# Benchamrk on paper  Depth Map Decomposition for Monocular Depth Estimation

1. Download InSpaceType eval set. Install [torch and torchivsion](https://pytorch.org/get-started/previous-versions/) and packages: matplotlib, pandas, opencv-python, tqdm, efficientnet_pytorch


2. Download pretrained model '51k_HRWSI.pth' from [Official Link](https://drive.google.com/drive/folders/1zsgT_5AO89WxzlFI53gwjomisb_Gkcox?usp=sharing) and put it here.

3.

  ```
  python demo.py --ckpt 51k_HRWSI.pth --filenames_file split_files.txt
  ```

  The command generates report files for hierarchy (H1-H3). *-all means overall H1-H3 means level of hierarchy. H1_xx means scene space type number. See [space_type_def.yml](https://github.com/DepthComputation/InSpaceType_Benchmark/blob/main/space_type_def.yml) for space type number definition.
