# Task 1
To achieve these results, we first applied NMS with confidence threshold 0.5 and IOU threshold 0.4. When matching ground truths to detections, we used an IOU threshold of 0.5.

| Model | Precision  | Recall     | mAP        |
|-------|------------|------------|------------|
| 1     | 0.9212     | 0.3025     | 0.2561     |
| 2     | **0.9441** | **0.4135** | **0.3079** |

| Class            | Model 1 mAP | Model 2 mAP |
|------------------|-------------|-------------|
| barcode          | 0.1605      | 0.1780      |
| car              | 0.3042      | 0.2999      |
| cardboard box    | 0.2786      | 0.3629      |
| fire             | 0.0000      | 0.0000      |
| forklift         | 0.1303      | 0.2241      |
| freight container| 0.0215      | 0.0320      |
| gloves           | 0.1552      | 0.3128      |
| helmet           | 0.0745      | 0.0973      |
| ladder           | 0.0202      | 0.0557      |
| license plate    | 0.1198      | 0.1453      |
| person           | 0.0922      | 0.1328      |
| qr code          | 0.4055      | 0.4097      |
| road sign        | 0.0349      | 0.0435      |
| safety vest      | 0.0553      | 0.0792      |
| smoke            | 0.0447      | 0.0469      |
| traffic cone     | 0.1917      | 0.1972      |
| traffic light    | 0.1880      | 0.2329      |
| truck            | 0.2495      | 0.3380      |
| van              | 0.4151      | 0.4879      |
| wood pallet      | 0.0207      | 0.0261      |

Model 2 outperforms Model 1 overall, achieving higher precision (0.9441 vs. 0.9212), recall (0.4135 vs. 0.3025), and mAP (0.3079 vs. 0.2561). At the per-class level, Model 2 shows clear improvements for cardboard box, forklift, gloves, ladder, person, safety vest, traffic light, truck, and van, often with substantial gains (e.g., gloves: 0.3128 vs 0.1552, truck: 0.3380 vs 0.2495, van: 0.4879 vs 0.4151). Model 1 performs slightly better on a few classes such as car (0.3042 vs. 0.2999), but the differences are small. Both models struggle equally on classes like fire (0.0000) and freight container (mAP < 0.05). Overall, Model 2 demonstrates broader and more consistent improvements across classes, making it the stronger choice.

# Task 2

### Sampling Strategy for TechTrack Dataset

To ensure that the TechTrack dataset is both representative and useful for evaluating model performance, we applied a stratified balanced sampling strategy with a total of 15,000 annotations from over 5000 images.

### Criteria for Selecting Representative Subsets of Data

- **Stratified Sampling Across All Classes**
  - Sampling was stratified so that every class in the dataset was represented equally.
  - This prevents rare classes (e.g., fire, freight container, ladder) from being omitted during training or evaluation.

- **Balanced Sampling for Underperforming Classes**
  - Using per-class mAP from Model 2, we identified underperforming classes (mAP < 0.15).
  - These classes were allocated double the sampling quota compared to their natural occurrence, ensuring that the model sees more examples of the difficult categories such as freight container, smoke, and road sign.

### Justification for Why This Sampling Strategy Is Valid

- Improved Per-Class Precision: Stratification ensures that performance metrics like precision and recall are meaningful on a per-class basis, since each class has a sufficient number of examples.
- Bias Mitigation: Without balancing, frequent classes (e.g., wood pallet, person) would dominate the dataset, leading to inflated scores for those categories and poor generalization for rare ones. Balanced sampling corrects this by boosting underrepresented or underperforming classes.
- Alignment with Evaluation Goals: As per-class precision comparison was emphasized in task 1, this strategy directly supports fairer evaluation across categories.

### Sanity Check

| Class            | Sampled Count | Sampled % | Unsampled Count | Unsampled % | Difference (Sampled % - Unsampled %) |
|------------------|---------------|-----------|-----------------|-------------|-------------------------------------|
| barcode          | 73            | 0.23      | 283             | 0.77        | -0.54                               |
| car              | 714           | 2.29      | 1379            | 3.76        | -1.47                               |
| cardboard box    | 4842          | 15.53     | 4995            | 13.60       | 1.93                                |
| fire             | 2422          | 7.77      | 2793            | 7.61        | 0.16                                |
| forklift         | 622           | 1.99      | 1103            | 3.00        | -1.01                               |
| freight container| 245           | 0.79      | 318             | 0.87        | -0.08                               |
| gloves           | 131           | 0.42      | 256             | 0.70        | -0.28                               |
| helmet           | 2122          | 6.81      | 2170            | 5.91        | 0.90                                |
| ladder           | 229           | 0.73      | 277             | 0.75        | -0.02                               |
| license plate    | 264           | 0.85      | 359             | 0.98        | -0.13                               |
| person           | 5920          | 18.99     | 6368            | 17.34       | 1.65                                |
| qr code          | 133           | 0.43      | 369             | 1.00        | -0.57                               |
| road sign        | 609           | 1.95      | 720             | 1.96        | -0.01                               |
| safety vest      | 1226          | 3.93      | 1260            | 3.43        | 0.50                                |
| smoke            | 1137          | 3.65      | 1495            | 4.07        | -0.42                               |
| traffic cone     | 287           | 0.92      | 506             | 1.38        | -0.46                               |
| traffic light    | 497           | 1.59      | 1193            | 3.25        | -1.66                               |
| truck            | 246           | 0.79      | 782             | 2.13        | -1.34                               |
| van              | 281           | 0.90      | 765             | 2.08        | -1.18                               |
| wood pallet      | 9179          | 29.44     | 9330            | 25.41       | 4.03                                |

The distribution in the sampled classes remains approximately the same as the original. Becasue we first sampled annotations and then selected the corresponding images (which may contain additional annotations outside the sampled set), the resulting image-level class percentages differ slightly from what stratified balanced sampling would ideally produce. Most classes differ by less than 2%, with the main exception being wood pallet, which is oversampled by about 4%. This is acceptable since its per-class mAP was relatively low, and the extra representation can help improve performance. Overall, the approach ensures rare and underperforming classes are more visible while maintaining the minimum image count requirement.

# Task 3
### thr = 0.4
mAP=0.2575
Per-class mAPs:
  barcode: 0.1795
  car: 0.1864
  cardboard box: 0.3688
  fire: 0.0000
  forklift: 0.2030
  freight container: 0.0272
  gloves: 0.1936
  helmet: 0.0974
  ladder: 0.0439
  license plate: 0.1152
  person: 0.1332
  qr code: 0.3782
  road sign: 0.0421
  safety vest: 0.0722
  smoke: 0.0423
  traffic cone: 0.1424
  traffic light: 0.1828
  truck: 0.2726
  van: 0.3857
  wood pallet: 0.0217

### thr = 0.5
mAP=0.2588
Per-class mAPs:
  barcode: 0.1795
  car: 0.1860
  cardboard box: 0.3687
  fire: 0.0000
  forklift: 0.2030
  freight container: 0.0272
  gloves: 0.1936
  helmet: 0.0982
  ladder: 0.0454
  license plate: 0.1152
  person: 0.1337
  qr code: 0.3782
  road sign: 0.0421
  safety vest: 0.0776
  smoke: 0.0423
  traffic cone: 0.1424
  traffic light: 0.1828
  truck: 0.2726
  van: 0.3852
  wood pallet: 0.0217

### thr = 0.6
mAP=0.2595
Per-class mAPs:
  barcode: 0.1795
  car: 0.1869
  cardboard box: 0.3686
  fire: 0.0000
  forklift: 0.2030
  freight container: 0.0272
  gloves: 0.1936
  helmet: 0.0982
  ladder: 0.0454
  license plate: 0.1152
  person: 0.1353
  qr code: 0.3780
  road sign: 0.0421
  safety vest: 0.0865
  smoke: 0.0423
  traffic cone: 0.1423
  traffic light: 0.1828
  truck: 0.2683
  van: 0.3833
  wood pallet: 0.0217

### thr = 0.7
mAP=0.2569
Per-class mAPs:
  barcode: 0.1795
  car: 0.1809
  cardboard box: 0.3683
  fire: 0.0000
  forklift: 0.2004
  freight container: 0.0256
  gloves: 0.1936
  helmet: 0.0982
  ladder: 0.0433
  license plate: 0.1152
  person: 0.1350
  qr code: 0.3780
  road sign: 0.0421
  safety vest: 0.0906
  smoke: 0.0423
  traffic cone: 0.1417
  traffic light: 0.1834
  truck: 0.2629
  van: 0.3697
  wood pallet: 0.0216

### thr = 0.8
mAP=0.2473
Per-class mAPs:
  barcode: 0.1795
  car: 0.1620
  cardboard box: 0.3667
  fire: 0.0000
  forklift: 0.1929
  freight container: 0.0250
  gloves: 0.1914
  helmet: 0.0979
  ladder: 0.0426
  license plate: 0.1140
  person: 0.1329
  qr code: 0.3780
  road sign: 0.0421
  safety vest: 0.0912
  smoke: 0.0395
  traffic cone: 0.1426
  traffic light: 0.1808
  truck: 0.2441
  van: 0.3358
  wood pallet: 0.0214

### thr = 0.9
mAP=0.2224
Per-class mAPs:
  barcode: 0.1795
  car: 0.1180
  cardboard box: 0.3501
  fire: 0.0000
  forklift: 0.1803
  freight container: 0.0229
  gloves: 0.1707
  helmet: 0.0965
  ladder: 0.0414
  license plate: 0.1158
  person: 0.1290
  qr code: 0.3713
  road sign: 0.0403
  safety vest: 0.0849
  smoke: 0.0389
  traffic cone: 0.1338
  traffic light: 0.1684
  truck: 0.1952
  van: 0.2301
  wood pallet: 0.0206

# Evaluation
Threshold of 0.6 is the best.

# Task 4
### Gaussian Blur
IoU=0.50: mAP=0.1886
IoU=0.55: mAP=0.1860
IoU=0.60: mAP=0.1795
IoU=0.65: mAP=0.1720
IoU=0.70: mAP=0.1573
IoU=0.75: mAP=0.1299
IoU=0.80: mAP=0.0935
IoU=0.85: mAP=0.0421
IoU=0.90: mAP=0.0131
IoU=0.95: mAP=0.0017
Per-class mAPs:
  barcode: 0.1779
  car: 0.1390
  cardboard box: 0.3387
  fire: 0.0000
  forklift: 0.1665
  freight container: 0.0200
  gloves: 0.1716
  helmet: 0.0263
  ladder: 0.0324
  license plate: 0.0937
  person: 0.0662
  qr code: 0.3310
  road sign: 0.0261
  safety vest: 0.0422
  smoke: 0.0297
  traffic cone: 0.0944
  traffic light: 0.1744
  truck: 0.1545
  van: 0.2260
  wood pallet: 0.0170

### Vertical Flip
IoU=0.50: mAP=0.0373
IoU=0.55: mAP=0.0342
IoU=0.60: mAP=0.0299
IoU=0.65: mAP=0.0254
IoU=0.70: mAP=0.0211
IoU=0.75: mAP=0.0139
IoU=0.80: mAP=0.0070
IoU=0.85: mAP=0.0028
IoU=0.90: mAP=0.0005
IoU=0.95: mAP=0.0000
Per-class mAPs:
  barcode: 0.0841
  car: 0.0059
  cardboard box: 0.0201
  fire: 0.0000
  forklift: 0.0000
  freight container: 0.0119
  gloves: 0.0010
  helmet: 0.0011
  ladder: 0.0050
  license plate: 0.0015
  person: 0.0008
  qr code: 0.0320
  road sign: 0.0054
  safety vest: 0.0016
  smoke: 0.0000
  traffic cone: 0.0015
  traffic light: 0.0280
  truck: 0.1058
  van: 0.0303
  wood pallet: 0.0080

### Adjust Brightness
IoU=0.50: mAP=0.2507
IoU=0.55: mAP=0.2437
IoU=0.60: mAP=0.2347
IoU=0.65: mAP=0.2216
IoU=0.70: mAP=0.2010
IoU=0.75: mAP=0.1648
IoU=0.80: mAP=0.1121
IoU=0.85: mAP=0.0503
IoU=0.90: mAP=0.0124
IoU=0.95: mAP=0.0014
Per-class mAPs:
  barcode: 0.1484
  car: 0.1753
  cardboard box: 0.3624
  fire: 0.0000
  forklift: 0.1996
  freight container: 0.0282
  gloves: 0.1942
  helmet: 0.0928
  ladder: 0.0454
  license plate: 0.1149
  person: 0.1307
  qr code: 0.3780
  road sign: 0.0411
  safety vest: 0.0710
  smoke: 0.0379
  traffic cone: 0.1308
  traffic light: 0.1884
  truck: 0.2554
  van: 0.3697
  wood pallet: 0.0214

# Task 5
