# Benchamrk on paper Adabins: Depth Estimation using adaptive bins

1. Download InSpaceType eval set. Install [torch and torchivsion](https://pytorch.org/get-started/previous-versions/) and packages: matplotlib, tqdm, pandas, opencv-python, tensorboardX

2. Download pretrained model 'AdaBins_nyu.pt' from [Official Link](https://drive.google.com/drive/folders/1nYyaQXOBjNdUJDsmJpcRpu6oE55aQoLA?usp=sharing) and put it under the folder 'pretrained'

3.

  ```
  python demo.py -i ../InSpaceType
  ```

  The command generates report files for hierarchy (H1-H3). *-all means overall H1-H3 means level of hierarchy. H1_xx means scene space type number. See [space_type_def.yml](https://github.com/DepthComputation/InSpaceType_Benchmark/blob/main/space_type_def.yml) for space type number definition.
