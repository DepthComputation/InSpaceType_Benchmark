# Benchamrk on paper  Iterative Refinement of Single-View Depth using Surface Normal and its Uncertainty

1. Download InSpaceType eval set. Install the requiremens.txt by '''pip install -r requirements.txt'''

2. Go to this [Official Link](https://drive.google.com/drive/folders/1idIVqOrJOK6kuidBng1K8sth-CyOfcCj?usp=sharing), and

* Download `*.pt` and place them under `./checkpoints`. Specifically the 'checkpoints' folder should include irondepth_* and normal_* four checkpoints


3.

  ```
   python test.py --train_data nyuv2 --test_data custom
  ```

  The command generates report files for hierarchy (H0-H2). *-all means overall H0-H2 means level of hierarchy. H1_xx means scene space type number. See [space_type_def.yml](https://github.com/DepthComputation/InSpaceType_Benchmark/blob/main/space_type_def.yml) for space type number definition.