# Benchamrk on paper Toward Practical Monocular Indoor Depth Estimation

1. Download InSpaceType eval set. Install [torch and torchivsion](https://pytorch.org/get-started/previous-versions/) and packages: matplotlib, tqdm, pandas, opencv-python, tensorboardX

2. Download pretrained model from [Official Link](https://drive.google.com/file/d/1kLJBuMOf0xSpYq7DtxnPpBTxMwW0ylGm/view?usp=sharing) and extract under 'ckpts-finetuned'. Speficially, ckpts-finetuned should contain encoder.pth and decoder.pth 

3.

  ```
  python demo.py
  ```
  
The command generates report files for hierarchy (H0-H2). *-all means overall H0-H2 means level of hierarchy. H1_xx means scene space type number. See [space_type_def.yml](https://github.com/DepthComputation/InSpaceType_Benchmark/blob/main/space_type_def.yml) for space type number definition.

