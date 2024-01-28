# Benchamrk on paper Global-Local Path Networks for Monocular Depth Estimation with Vertical CutDepth

1. Download InSpaceType eval set. Install [torch and torchivsion](https://pytorch.org/get-started/previous-versions/) and packages: matplotlib, pandas, transformers, opencv-python, tqdm 

2.

  ```
  python demo_glpn.py -i ../InSpaceType
  ```

  The command generates report files for hierarchy (H1-H3). *-all means overall H1-H3 means level of hierarchy. H1_xx means scene space type number. See [space_type_def.yml](https://github.com/DepthComputation/InSpaceType_Benchmark/blob/main/space_type_def.yml) for space type number definition.
