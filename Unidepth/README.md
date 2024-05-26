# Benchamrk on paper Unidepth

1. Download InSpaceType eval set and put data under 'InspaceType' under the root. 

InSpaceType_Benchmark
  | - InSpaceType
            |- 0001.pfm
            |- 0001_L.jpg
              ....
  | - Method 1
  | - Method 2
    ......

Install [torch and torchivsion](https://pytorch.org/get-started/previous-versions/) and packages: matplotlib, tqdm, opencv-python, xFormers

2.

  ```
   python demo.py --img-path ../InSpaceType --outdir ./vis_depth
  ```

  The command generates report files for hierarchy (H0-H2). *-all means overall H0-H2 means level of hierarchy. H1_xx means scene space type number. See [space_type_def.yml](https://github.com/DepthComputation/InSpaceType_Benchmark/blob/main/space_type_def.yml) for space type number definition.
  Colored visualization in metric depth are saved under --outdir