# Benchamrk on paper ZoeDepth

1. Download InSpaceType eval set. Install [torch and torchivsion](https://pytorch.org/get-started/previous-versions/) and packages: matplotlib, tqdm, tensorboardX, timm, mmcv, opencv-python

2.

  ```  
   python demo.py -i split_files.txt -o outputs/
  ```

  The command generates report files for hierarchy (H1-H3). *-all means overall H1-H3 means level of hierarchy. H1_xx means scene space type number. See [space_type_def.yml](https://github.com/DepthComputation/InSpaceType_Benchmark/blob/main/space_type_def.yml) for space type number definition.
