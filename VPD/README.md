# VPD

1. Follow [VPD] (https://github.com/wl-zhao/VPD) Installation section (download stable diffusion models and install stable-diffusion package). The VPD repo uses SD v1-5

2. Follow [VPD depth](https://github.com/wl-zhao/VPD/blob/main/depth/README.md) first step to install mmcv and requirements. Then download [VPD depth pretrained](https://cloud.tsinghua.edu.cn/f/7e4adc76cc9b4200ac79/?dl=1) and put it under checkpoints/

3. Download InSpaceType eval set and put it under root folder.

4. 

  ```
  cd depth

  bash test.sh ../checkpoints/vpd_depth_480x480.pth
  ```
  
  The command generates report files for hierarchy (H0-H2). *-all means overall H0-H2 means level of hierarchy. H1_xx means scene space type number. See [space_type_def.yml](https://github.com/DepthComputation/InSpaceType_Benchmark/blob/main/space_type_def.yml) for space type number definition. 