# <div align=""> InSpaceType: Reconsider Space Type in Indoor Monocular Depth Estimation </div>

This repository includes codes for reproducing benchmark reuslts for the paper.


## <div align="">Data</div>

[Sample data](https://drive.google.com/file/d/1ePsiverqYofCwuZJv98tLPWSj8bNU3ne/view?usp=sharing): This contains 167MB sample data

[InSpaceType Eval set](https://drive.google.com/file/d/1d3DiLPVEEk-hRvhaEfSK6adu5DPBdlF-/view?usp=sharing): This contains ~11.5G evaluation set

For evaluation, please donwload the eval set, extract under this root folder and rename it to 'InSpaceType'

Speficially, the data structure should be

```
InSpaceType
        |---- 0001.pfm
        |---- 0001_L.jpg
        |---- 0002.pfm
        |---- 0002_L.jpg
        |---- 0003.pfm
        |---- 0003_L.jpg
        |---- ...
```

Then go to each subfolder and see respective README instruction for evalution.


[InSpaceType all data](): This contains about 500G the whole InSpaceType dataset

The data is indexed by <seq_num> folders. In each folder, it contains images and depth maps.

## <div align="">Benchmark</div>

Overall benchmark
<img src='pics/overall.png'>

Sample spaceType breakdown
<img src='pics/type.png'>

sample heirarchy labeling and breakdown
<img src='pics/heirarchy.png'>


Please refer to the paper and the supplementary for the full results.

## License
The dataset is CC BY-SA 4.0 licensed.
