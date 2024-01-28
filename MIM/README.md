# Benchamrk on paper Revealing the Dark Secrets of Masked Image Modeling (Depth Estimation)

1. Download InSpaceType eval set. Install the requiremens.txt by '''pip install -r requirements.txt'''

2. Download pretrained model 'nyudepthv2_swin_large.ckpt' from [Official Link](https://mailustceducn-my.sharepoint.com/:f:/g/personal/aa397601_mail_ustc_edu_cn/EkoYQyhiD6hJu9CGYLOwiF8BRqHgk8kX61NUcyfmdOUV7Q?e=h2uctw) and put it under the folder 'ckpt'

3.

  ```
  python test.py --dataset nyudepthv2 --data_path ../data/ --max_depth 10.0 --max_depth_eval 10.0  --backbone swin_large_v2 --depths 2 2 18 2 --num_filters 32 32 32 --deconv_kernels 2 2 2 --window_size 30 30 30 15 --pretrain_window_size 12 12 12 6 --use_shift True True False False --flip_test --shift_window_test --shift_size 2 --do_evaluate --ckpt_dir ckpt/nyudepthv2_swin_large.ckpt 
  ```

  The command generates report files for hierarchy (H1-H3). *-all means overall H1-H3 means level of hierarchy. H1_xx means scene space type number. See [space_type_def.yml](https://github.com/DepthComputation/InSpaceType_Benchmark/blob/main/space_type_def.yml) for space type number definition.