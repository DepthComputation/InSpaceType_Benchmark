# Benchamrk on paper Vision Transformers for Dense Prediction

1. Download InSpaceType eval set. Install the requiremens.txt by '''pip install -r requirements.txt'''

2. Download pretrained model 'dpt_hybrid_nyu-2ce69ec7.pt' from [Official Link](https://github.com/intel-isl/DPT/releases/download/1_0/dpt_hybrid_nyu-2ce69ec7.pt) and put it under the folder 'weights'

3.

  ```
  python run_monodepth.py -t dpt_hybrid_nyu -i ../InSpaceType
  ```

  The command generates report files for hierarchy (H0-H2). *-all means overall H0-H2 means level of hierarchy. H1_xx means scene space type number. See [space_type_def.yml](https://github.com/DepthComputation/InSpaceType_Benchmark/blob/main/space_type_def.yml) for space type number definition.
