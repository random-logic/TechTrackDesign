# Task 1
First, we calculated the precision and recall for both models at confidence threshold 0.5 and IOU threshold 0.4.

| Model | Precision | Recall |
|-------|-----------|--------|
| 1     | 0.9212    | 0.3025 |
| 2     | 0.9441    | 0.4135 |

Per-class mAPs:
  barcode: 0.1780
  car: 0.2999
  cardboard box: 0.3629
  fire: 0.0000
  forklift: 0.2241
  freight container: 0.0320
  gloves: 0.3128
  helmet: 0.0973
  ladder: 0.0557
  license plate: 0.1453
  person: 0.1328
  qr code: 0.4097
  road sign: 0.0435
  safety vest: 0.0792
  smoke: 0.0469
  traffic cone: 0.1972
  traffic light: 0.2329
  truck: 0.3380
  van: 0.4879
  wood pallet: 0.0261

For both precision and recall, model 2 appears to do a better job.

# Task 2
We can use stratified sampling to ensure that the model accounts for all classes equally.

Class distribution in sampled dataset:
barcode: 68 (0.45%)
car: 334 (2.23%)
helmet: 1053 (7.02%)
person: 3090 (20.61%)
license plate: 174 (1.16%)
road sign: 349 (2.33%)
truck: 189 (1.26%)
van: 185 (1.23%)
ladder: 134 (0.89%)
traffic light: 289 (1.93%)
traffic cone: 122 (0.81%)
forklift: 267 (1.78%)
cardboard box: 1212 (8.09%)
qr code: 89 (0.59%)
gloves: 62 (0.41%)
safety vest: 611 (4.08%)
wood pallet: 4528 (30.21%)
freight container: 154 (1.03%)
fire: 1355 (9.04%)
smoke: 725 (4.84%)

Class distribution in unsampled dataset:
car: 1379 (3.76%)
truck: 782 (2.13%)
barcode: 283 (0.77%)
van: 765 (2.08%)
person: 6368 (17.34%)
helmet: 2170 (5.91%)
safety vest: 1260 (3.43%)
traffic cone: 506 (1.38%)
cardboard box: 4995 (13.60%)
fire: 2793 (7.61%)
traffic light: 1193 (3.25%)
road sign: 720 (1.96%)
gloves: 256 (0.70%)
wood pallet: 9330 (25.41%)
forklift: 1103 (3.00%)
smoke: 1495 (4.07%)
freight container: 318 (0.87%)
qr code: 369 (1.00%)
license plate: 359 (0.98%)
ladder: 277 (0.75%)

# Task 3
### thr = 0.4
IoU=0.50: mAP=0.2575
IoU=0.55: mAP=0.2491
IoU=0.60: mAP=0.2411
IoU=0.65: mAP=0.2305
IoU=0.70: mAP=0.2113
IoU=0.75: mAP=0.1715
IoU=0.80: mAP=0.1134
IoU=0.85: mAP=0.0531
IoU=0.90: mAP=0.0150
IoU=0.95: mAP=0.0015
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
IoU=0.50: mAP=0.2588
IoU=0.55: mAP=0.2499
IoU=0.60: mAP=0.2415
IoU=0.65: mAP=0.2308
IoU=0.70: mAP=0.2116
IoU=0.75: mAP=0.1718
IoU=0.80: mAP=0.1136
IoU=0.85: mAP=0.0532
IoU=0.90: mAP=0.0150
IoU=0.95: mAP=0.0015
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
IoU=0.50: mAP=0.2595
IoU=0.55: mAP=0.2506
IoU=0.60: mAP=0.2422
IoU=0.65: mAP=0.2315
IoU=0.70: mAP=0.2115
IoU=0.75: mAP=0.1717
IoU=0.80: mAP=0.1136
IoU=0.85: mAP=0.0531
IoU=0.90: mAP=0.0149
IoU=0.95: mAP=0.0015
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
IoU=0.50: mAP=0.2569
IoU=0.55: mAP=0.2477
IoU=0.60: mAP=0.2390
IoU=0.65: mAP=0.2296
IoU=0.70: mAP=0.2094
IoU=0.75: mAP=0.1711
IoU=0.80: mAP=0.1139
IoU=0.85: mAP=0.0523
IoU=0.90: mAP=0.0147
IoU=0.95: mAP=0.0015
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
IoU=0.50: mAP=0.2473
IoU=0.55: mAP=0.2387
IoU=0.60: mAP=0.2304
IoU=0.65: mAP=0.2215
IoU=0.70: mAP=0.2022
IoU=0.75: mAP=0.1679
IoU=0.80: mAP=0.1144
IoU=0.85: mAP=0.0517
IoU=0.90: mAP=0.0146
IoU=0.95: mAP=0.0014
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
IoU=0.50: mAP=0.2224
IoU=0.55: mAP=0.2139
IoU=0.60: mAP=0.2069
IoU=0.65: mAP=0.1982
IoU=0.70: mAP=0.1809
IoU=0.75: mAP=0.1500
IoU=0.80: mAP=0.1047
IoU=0.85: mAP=0.0506
IoU=0.90: mAP=0.0148
IoU=0.95: mAP=0.0015
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
