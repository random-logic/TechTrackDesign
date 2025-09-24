# %%
import os

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def get_path_in_parent(*args):
    return os.path.abspath(os.path.join(os.getcwd(), '..', *args))

def get_path_in_storage(*args):
    return get_path_in_parent("storage", *args)

def get_logistics_path(*args):
    return get_path_in_storage("logistics", *args)

categories = [{'id': 0, 'name': 'barcode'},
              {'id': 1, 'name': 'car'},
              {'id': 2, 'name': 'cardboard box'},
              {'id': 3, 'name': 'fire'},
              {'id': 4, 'name': 'forklift'},
              {'id': 5, 'name': 'freight container'},
              {'id': 6, 'name': 'gloves'},
              {'id': 7, 'name': 'helmet'},
              {'id': 8, 'name': 'ladder'},
              {'id': 9, 'name': 'license plate'},
              {'id': 10, 'name': 'person'},
              {'id': 11, 'name': 'qr code'},
              {'id': 12, 'name': 'road sign'},
              {'id': 13, 'name': 'safety vest'},
              {'id': 14, 'name': 'smoke'},
              {'id': 15, 'name': 'traffic cone'},
              {'id': 16, 'name': 'traffic light'},
              {'id': 17, 'name': 'truck'},
              {'id': 18, 'name': 'van'},
              {'id': 19, 'name': 'wood pallet'}]

# Load them into COCO API
coco_gt = COCO(get_path_in_storage("gts_coco_1.json"))
coco_dt = coco_gt.loadRes(get_path_in_storage("pred_coco_1.json"))

# Run evaluation
coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()

# Precision and recall arrays
precisions = coco_eval.eval['precision']  # shape: [IoU, recall, cls, area, maxDets]
recalls = coco_eval.eval['recall']        # shape: [IoU, cls, area, maxDets]

print("Precision array shape:", precisions.shape)
print("Recall array shape:", recalls.shape)

# Compute and print per-IoU-threshold mAP values
iou_thrs = coco_eval.params.iouThrs
per_iou_map = []
for iou_idx, iou_thr in enumerate(iou_thrs):
    # For each IoU threshold, get the precision values: shape [recall, cls, area, maxDets]
    prec = precisions[iou_idx, :, :, 0, 2]  # area=0 (all), maxDets=2 (100)
    # Only consider valid values (>-1)
    valid = prec[prec > -1]
    if valid.size > 0:
        mean_ap = np.mean(valid)
    else:
        mean_ap = float('nan')
    per_iou_map.append(mean_ap)
    print(f"IoU={iou_thr:.2f}: mAP={mean_ap:.4f}")

# Compute and print per-class mAPs averaged over all IoU thresholds and recall values
num_classes = len(categories)
per_class_map = []
for cls_idx in range(num_classes):
    # Extract precision for all IoU thresholds and all recall values for this class
    # shape: [IoU, recall]
    class_precisions = precisions[:, :, cls_idx, 0, 2]  # area=0, maxDets=2
    valid = class_precisions[class_precisions > -1]
    if valid.size > 0:
        class_map = np.mean(valid)
    else:
        class_map = float('nan')
    per_class_map.append(class_map)

print("Per-class mAPs:")
for cls_idx, class_info in enumerate(categories):
    class_name = class_info['name']
    print(f"  {class_name}: {per_class_map[cls_idx]:.4f}")

# %%
