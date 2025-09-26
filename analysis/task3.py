# %%
from typing import Dict, List, Tuple
import h5py
import os
import cv2
import numpy as np

def get_path_in_parent(*args):
    return os.path.abspath(os.path.join(os.getcwd(), '..', *args))

def get_path_in_storage(*args):
    return get_path_in_parent("storage", *args)

def get_logistics_path(*args):
    return get_path_in_storage("logistics", *args)

def list_logistics_dir():
    return os.listdir(get_logistics_path())

def get_nms(pred: Dict[str, List[Tuple[int, int, int, int, float, int]]],
            score_threshold=0.5,
            nms_threshold=0.4) -> Dict[str, List[Tuple[int, int, int, int, float, int]]]:
    res = {}
    for key, dets in pred.items():
        bboxes = [list(det[:4]) for det in dets]
        scores = [det[4] for det in dets]

        indices = cv2.dnn.NMSBoxes(bboxes, scores, score_threshold, nms_threshold)

        # normalize indices into a flat list
        if indices is None or len(indices) == 0:
            res[key] = []
        else:
            indices = np.array(indices).flatten().tolist()
            res[key] = [dets[i] for i in indices]

    return res

def load_gts_from_h5(
    model_num: int
) -> Dict[str, List[Tuple[int, int, int, int, int]]]:
    """
    Load ground truths from an HDF5 file.

    Returns a dictionary mapping string keys to lists of tuples (class_id, x, y, w, h)
    x,y,w,h are all int absolute value pixel coords
    """
    gts = {}
    with h5py.File(get_path_in_storage(f"gts{model_num}_sampled.h5"), "r") as f:
        for key in f.keys():
            arr = f[key][()]
            gts[key] = [(int(row[0]), int(row[1]), int(row[2]), int(row[3]), int(row[4])) for row in arr]
    return gts

def load_detections_from_h5(
    model_num: int
) -> Dict[str, List[Tuple[int, int, int, int, float, int]]]:
    """
    Load detections from an HDF5 file.
    model_num: int - the model num to use

    Returns a dictionary mapping string keys to lists of tuples (x, y, w, h, confidence, class_id) where x, y, w, h are all absolute pixel values
    """
    detections = {}
    with h5py.File(get_path_in_storage(f"out{model_num}_sampled.h5"), "r") as f:
        for key in f.keys():
            arr = f[key][()]
            detections[key] = [(int(row[0]), int(row[1]), int(row[2]), int(row[3]), float(row[4]), int(row[5])) for row in arr]
    return detections

# %%
out2 = load_detections_from_h5(2)
gts = load_gts_from_h5(2)

# %%
# Get Coco formats
def get_coco_gts_and_preds(gts: Dict[str, List[Tuple[int, int, int, int, int]]], preds: Dict[str, List[Tuple[int, int, int, int, float, int]]], img_dims = (640, 640)) -> Tuple[Dict, List]:
    imgs = []
    annotations = []
    pred_res = []

    for img_id, (img_name, img_gts) in enumerate(gts.items()):
        imgs.append({
            "id": img_id,
            "file_name": f"{img_name}.jpg",
            "width": img_dims[0],
            "height": img_dims[1]
        })

        for class_id, x, y, w, h in img_gts:
            annotations.append({
                "id": len(annotations),
                "image_id": img_id,
                "category_id": class_id,
                "bbox": [x, y, w, h],
                "area": w * h,
                "iscrowd": 0
            })

        for x, y, w, h, conf_score, class_id in preds[img_name]:
            pred_res.append({
                "id": len(pred_res),
                "image_id": img_id,
                "category_id": class_id,
                "bbox": [x, y, w, h],
                "score": conf_score
            })

    return {
        "info": {},
        "licenses": [],
        "images": imgs,
        "annotations": annotations,
        "categories": [
            {'id': 0, 'name': 'barcode'},
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
            {'id': 19, 'name': 'wood pallet'}
        ]
    }, pred_res

# %%
# Run evaluation
import json
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def evaluate(thr: float):
    pred_nms = get_nms(out2, 0.5, thr)
    gts_coco, pred_coco_nms = get_coco_gts_and_preds(gts, pred_nms)
    with open(get_path_in_storage("gts_coco_nms.json"), "w") as f:
        json.dump(gts_coco, f)
    with open(get_path_in_storage("pred_coco_nms.json"), "w") as f:
        json.dump(pred_coco_nms, f)

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
    coco_gt = COCO(get_path_in_storage("gts_coco_nms.json"))
    coco_dt = coco_gt.loadRes(get_path_in_storage("pred_coco_nms.json"))

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
evaluate(0.4)

# %%
evaluate(0.5)

# %%
evaluate(0.6)

# %%
evaluate(0.7)

# %%
evaluate(0.8)

# %%
evaluate(0.9)

# %%
