# <div align=""> InSpaceType: Reconsider Space Type in Indoor Monocular Depth Estimation </div>

This repository includes codes for reproducing benchmark reuslts for the paper.


## <div align="">Data</div>

[Sample data](https://drive.google.com/file/d/1ePsiverqYofCwuZJv98tLPWSj8bNU3ne/view?usp=sharing): This contains 167MB sample data

[InSpaceType Eval set](https://drive.google.com/file/d/1d3DiLPVEEk-hRvhaEfSK6adu5DPBdlF-/view?usp=sharing): This contains 1260 RGBD pairs for evaluation use about 11.5G

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


[InSpaceType all data](https://drive.google.com/drive/folders/1EjdInytpvYWBT3BmQIDsTzFyP0dngP1U?usp=sharing): This contains 40K RGBD pairs, about 500G the whole InSpaceType dataset. The whole data is split into 8 chunks. Please download all chunks in the folder and extract them.

The data is indexed by <seq_num> folders. In each folder, it contains images and depth maps.


<font size="30"> Analysis I-II [Benchmark on overall performance and space type breakdown]: </font>
The table shows challenging InSpaceType benchmark for the overall performance, following the major monocular depth estimation protocol and metrics. We adopt the following publicly released models trained on NYUv2 for evaluation. Recent work using larger-scale backbone models obtains lower error and higher accuracy.
<img src='pics/overall.png'>

Sample SpaceType breakdown is shown as follows. Different from conventional monocular depth estimation protocols, our work pioneers to study space type breakdown for a method. This provides a hint on how a method performs across different space types. From the following table one can observe the state-of-the-art models suffer from major performance imbalance issue. Both methods have similar easy and hard type which are potentially inherited from NYUv2. Directly deploying those SOTA models for in-the-wild application can add robustness concern.
<img src='pics/type.png'>

<font size="30"> Analysis III [Training datasets]: </font>
In addition to NYUv2, we analyze other popular training datasets: Hypersim, SimSIN, UniSIN for indoor moncular depth estimation. One can find models trained on each training dataset also suffer from imbalance between space types, revealing their underlying bias. We also find that kitchen is a special type with lower RMSE but also very low accuracy. We assume this is because kitchen contains many cluttered small objects, such as bottles, kitchenware, and utensils in the near field. Current synthetic datasets (SimSIN, Hypersim) may not attain the same level of simulation complexity and thus training on them may not match the real-world complexity.
<img src='pics/dataset-1.png'>
<img src='pics/dataset-2.png'>

<font size="30"> Analysis IV-V [Dataset fitting and bias mitigation]: </font>
We first creat a larger InSpaceType training set and study a dataset fitting problem. This aims to show how each space type fits when training all the types together and attempts to observe concordance between types. From the table large room and lounge are large-size spaces and naturally result in slightly higher RMSE. It is worth noting that there is an apparent trend: for errors, larger rooms and longer ranges tend to have a higher estimation error; for accuracy, arbitrarily arranged small objects in the near field are challenging, a frequent scenario for kitchen.
<img src='pics/fitting.png'>

<font size="30"> Analysis IV-V [Dataset fitting and bias mitigation]: </font>
We then study three different strategies to alleviate type imbalanceL meta-learning (ML), class reweighting (CR), and class-balance sampler (CBS). One can find CBS and ML are better strategies to attain lower standard deviation across types (t-STD) and better overall performance. Though CR attains lower t-STD, its overall performance drop as well. This is because CR could harm head-class performances as observed in literature.
<img src='pics/mitigation.png'>


<font size="30"> Analysis VI [Intra-group generalization] </font>
We next investigate generalization to unseen types. We divide the whole InSpaceType training set into different splits, train on each division, and then evaluate on InSpaceType eval split. The whole training set is divided into three groups based on similarity between types and concerns a situation where one collects training data almost in the same functionality that matches the primary application scenarios without considering different user scenarios. The left half shows generalization to other types, and the right half shows evaluation on different depth ranges. Training on specific groups can produce good performance on its dedicated types. However, one can observe training on only some types encounters severe issues in generalization to other unseen types, which further reveal high variation between different indoor environments, and pretrained knowledge on some types may not easily transfer to other types.
<img src='pics/group.png'>

<font size="30"> Conclusion </font>
Unlike previous methods that focus on algorithmic developments, we are the first work to consider space types in indoor monocular depth estimation for robustness and practicability in deployment. We point out limitations in previous evaluations where performance variances across types are overlooked and present a novel dataset, InSpaceType, along with a hierarchical space type definition to facilitate our study. We give thorough studies to analyze and benchmark performance based on space types. Ten high-performing methods are examined, and we find they suffer from severe performance imbalance between space types. We analyze a total of 4 training datasets and enumerate their strength and weakness space types. 3 popular strategies, namely, class reweighting, type-balanced sampler, and meta-learning, are studied to mitigate imbalance. Further, we find generalization to unseen space types challenging due to high diversity of objects and mismatched scales across types. Overall, this work pursues a practical purpose and emphasizes the importance of this usually overlooked factor- space type in indoor environments. We call for attention to safety concerns for model deployment without considering performance variance across space types.
<img src='pics/group.png'>

<font size="30"> Sample heirarchy labeling and breakdown </font>
<img src='pics/heirarchy.png'>


Please refer to the paper and the supplementary for the full results.

## License
The dataset is CC BY-SA 4.0 licensed.
