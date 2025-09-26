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
