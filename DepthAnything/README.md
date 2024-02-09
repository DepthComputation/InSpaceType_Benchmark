# Benchamrk on paper Depth-Anything

1. Download InSpaceType eval set. Install [torch and torchivsion](https://pytorch.org/get-started/previous-versions/) and packages: matplotlib, tqdm, opencv-python

2. Download Depth-Anything NYUv2 finetuned model (depth_anything_metric_depth_indoor.pt) from [official link](https://huggingface.co/spaces/LiheYoung/Depth-Anything/tree/main/checkpoints_metric_depth) and place it under 'metric_depth/checkpoints'

3.

  ```
   cd metric_depth

   python demo.py --img-path ../../InSpaceType --outdir ./vis_depth
  ```

  The command generates report files for hierarchy (H0-H2). *-all means overall H0-H2 means level of hierarchy. H1_xx means scene space type number. See [space_type_def.yml](https://github.com/DepthComputation/InSpaceType_Benchmark/blob/main/space_type_def.yml) for space type number definition.
  Colored visualization in metric depth are saved under --outdir
