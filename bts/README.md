# Benchamrk on From Big to Small: Multi-Scale Local Planar Guidance for Monocular Depth Estimation   

1. Download InSpaceType eval set. Install [torch and torchivsion](https://pytorch.org/get-started/previous-versions/) and packages: matplotlib, tqdm, pandas, opencv-python, tensorboardX

2. Download pretrained model 'bts_nyu_v2_pytorch_densenet161.zip' from [Official Link](https://cogaplex-bts.s3.ap-northeast-2.amazonaws.com/bts_nyu_v2_pytorch_densenet161.zip) and extract under 'model'

3.
  cd pytorch
  
  ```
  python bts_test.py  --dataset nyu --filenames_file ../train_test_inputs/split_files.txt --checkpoint_path models/bts_nyu_v2_pytorch_densenet161/model --max_depth 10 --encoder densenet161_bts --model_name bts_nyu_v2_pytorch_densenet161
  ```

