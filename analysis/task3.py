#!/usr/bin/env python
# coding: utf-8

# In[5]:


import os
import h5py
import numpy as np
from typing import Dict, Tuple


def get_path_in_parent(*args):
    return os.path.abspath(os.path.join(os.getcwd(), '..', *args))

def get_path_in_storage(*args):
    return get_path_in_parent("storage", *args)

def get_logistics_path(*args):
    return get_path_in_storage("logistics", *args)

def save_outputs(model_num: int, outputs: Dict[str, Tuple[np.ndarray, ...]], nms: bool = False) -> None:
    """Save a dict of tuples of arrays to an HDF5 file."""
    path = get_path_in_storage(f"outputs_{model_num}{"_nms" if nms else ""}.h5")
    with h5py.File(path, "w") as f:
        for key, tup in outputs.items():
            grp = f.create_group(str(key))
            for i, arr in enumerate(tup):
                grp.create_dataset(
                    f"array_{i}", data=arr, compression="gzip", compression_opts=1
                )

def load_outputs(model_num: int, nms: bool = False) -> Dict[str, Tuple[np.ndarray, ...]]:
    """Load a dict of tuples of arrays from an HDF5 file."""
    path = get_path_in_storage(f"outputs_{model_num}{"_nms" if nms else ""}.h5")
    outputs_loaded = {}
    with h5py.File(path, "r") as f:
        for key in f.keys():
            grp = f[key]
            arrays = tuple(np.array(grp[subkey]) for subkey in sorted(grp.keys()))
            outputs_loaded[key] = arrays
    return outputs_loaded


# In[4]:


outputs_2: Dict[str, Tuple[np.ndarray, ...]] = load_outputs(2)


# In[7]:


import cv2
import numpy as np
from multiprocessing import Pool
from typing import Dict, Tuple

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def apply_nms(outputs: Dict[str, Tuple[np.ndarray, ...]],
              conf_threshold: float = 0.5,
              nms_threshold: float = 0.4
             ) -> Dict[str, Tuple[np.ndarray, ...]]:
    """
    Apply Non-Max Suppression (NMS) to YOLO-style outputs.

    Args:
        outputs (dict): filename -> raw outputs from get_outputs()
        conf_threshold (float): confidence threshold
        nms_threshold (float): IoU threshold for NMS

    Returns:
        dict: filename -> tuple of np.ndarrays, same format as net.forward
    """
    filtered_outputs = {}

    for fname, layer_outputs in outputs.items():
        detections = []

        for out in layer_outputs:  # out shape: (N, 85)
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = detection[4] * scores[class_id]

                if confidence > conf_threshold:
                    cx, cy, w, h = detection[0:4]
                    x = cx - w / 2
                    y = cy - h / 2

                    # build YOLO-style detection row
                    det_row = np.zeros_like(detection)
                    det_row[0:4] = [x, y, w, h]
                    det_row[4] = detection[4]          # objectness
                    det_row[5:] = 0.0
                    det_row[5 + class_id] = scores[class_id]  # keep only top class

                    detections.append((det_row, confidence))

        if not detections:
            filtered_outputs[fname] = (np.zeros((0, 85), dtype=np.float32),)
            continue

        # unpack for NMS
        boxes = []
        confidences = []
        det_rows = []
        for det_row, conf in detections:
            x, y, w, h = det_row[0:4]
            boxes.append([int(x), int(y), int(w), int(h)])
            confidences.append(float(conf))
            det_rows.append(det_row)

        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

        if len(indices) > 0:
            final = np.array([det_rows[i] for i in indices.flatten()], dtype=np.float32)
        else:
            final = np.zeros((0, 85), dtype=np.float32)

        # wrap in tuple (to mimic net.forward output structure)
        filtered_outputs[fname] = (final,)

    return filtered_outputs

def compute_map(preds, gt_json, iou_type="bbox"):
    """
    Compute mAP using pycocotools.

    Args:
        preds (list of dict): predictions in COCO format
        gt_json (str): path to ground-truth annotations in COCO format
        iou_type (str): "bbox" or "segm"

    Returns:
        float: mAP@[0.5:0.95]
    """
    coco_gt = COCO(gt_json)
    coco_dt = coco_gt.loadRes(preds)

    coco_eval = COCOeval(coco_gt, coco_dt, iouType=iou_type)
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # By convention, coco_eval.stats[0] = mAP@[.5:.95]
    return coco_eval.stats[0]

def evaluate_threshold(nms_thresh: float,
                       outputs: Dict[str, Tuple[np.ndarray, ...]],
                       gts: Dict[str, np.ndarray],  # filename -> ground truth boxes
                       conf_threshold: float = 0.5) -> Tuple[float, float]:
    """
    Apply NMS with given threshold and compute mAP.
    Returns (nms_thresh, mAP).
    """
    # Apply NMS
    preds = apply_nms(outputs, conf_threshold=conf_threshold, nms_threshold=nms_thresh)

    # Compute mAP (user must implement this)
    map_score = compute_map(preds, gts)

    return nms_thresh, map_score

def get_ground_truths() -> Dict[str, np.ndarray]:
    logistics_path = get_path_in_storage("logistics")
    res = {}

    for filename in os.listdir(logistics_path):
        if filename.endswith(".txt"):
            filepath = os.path.join(logistics_path, filename)

            with open(filepath, "r") as f:
                lines = []

                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:  # expecting id, x, y, width, height
                        t = (
                            int(parts[0]),
                            float(parts[1]),
                            float(parts[2]),
                            float(parts[3]),
                            float(parts[4]),
                        )
                        lines.append(t)

                res[filename[:-4]] = np.array(lines)

    return res


# In[9]:


import os
import json
from PIL import Image

def convert_gts_to_coco_format(gts: Dict[str, np.ndarray],
                               images_path: str,
                               class_names=None,
                               start_ann_id=1):
    """
    Convert YOLO-style ground truths to COCO format dict.

    Args:
        gts (dict): filename (no ext) -> np.array of [class_id, x, y, w, h]
        images_path (str): path to folder with images (to get sizes)
        class_names (list, optional): class names for categories
        start_ann_id (int): starting annotation id

    Returns:
        dict: COCO-style annotations {images, annotations, categories}
    """
    coco = {
        "info": {
            "description": "Custom Dataset",
            "version": "1.0",
            "year": 2025,
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": []
    }
    ann_id = start_ann_id

    # Build categories
    num_classes = max(int(gt[0]) for arr in gts.values() for gt in arr) + 1
    if class_names:
        coco["categories"] = [{"id": i + 1, "name": name} for i, name in enumerate(class_names)]
    else:
        coco["categories"] = [{"id": i + 1, "name": f"class_{i}"} for i in range(num_classes)]

    # Loop over images
    for img_id, (fname, anns) in enumerate(gts.items(), start=1):
        img_file = os.path.join(images_path, f"{fname}.jpg")
        with Image.open(img_file) as img:
            w_img, h_img = img.size

        coco["images"].append({
            "id": img_id,
            "file_name": f"{fname}.jpg",
            "width": w_img,
            "height": h_img
        })

        if anns.size == 0:
            continue

        # Loop over annotations
        for gt in anns:
            cls_id, x, y, w, h = gt

            # Convert from normalized center format → COCO top-left format
            x_min = (x - w / 2) * w_img
            y_min = (y - h / 2) * h_img
            width = w * w_img
            height = h * h_img

            coco["annotations"].append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": int(cls_id) + 1,
                "bbox": [x_min, y_min, width, height],
                "area": width * height,
                "iscrowd": 0
            })
            ann_id += 1

    return coco


# In[10]:


gts = get_ground_truths()
coco_gt = convert_gts_to_coco_format(gts, get_path_in_storage("logistics"))


# In[11]:


coco_gt


# In[15]:


import os
from PIL import Image
from typing import Dict, Tuple
import numpy as np

def convert_to_coco_format(preds: Dict[str, Tuple[np.ndarray, ...]],
                           images_path: str,
                           class_names=None,
                           image_id_map=None) -> list:
    """
    Convert YOLO-style predictions (after NMS) into COCO format.
    preds: Dict[filename, Tuple[np.ndarray, ...]]
    """
    coco_results = []
    ann_id = 1

    for filename, detections_tuple in preds.items():
        # Each entry is a tuple of np.ndarrays -> concatenate
        if detections_tuple is None or all(len(arr) == 0 for arr in detections_tuple):
            continue

        detections = np.vstack([arr for arr in detections_tuple if arr is not None and len(arr) > 0])

        if detections.size == 0:
            continue

        # get true image size
        img_file = os.path.join(images_path, f"{filename}.jpg")
        with Image.open(img_file) as img:
            w_img, h_img = img.size

        image_id = image_id_map[filename] if image_id_map else abs(hash(filename)) % (10**6)

        for det in detections:
            # Expecting det = [x_center, y_center, w, h, score, class_id]
            if len(det) < 6:
                raise ValueError(f"Detection format error for {filename}: got {det}")

            x_center, y_center, w, h, score, cls_id = det[:6]

            # scale normalized coords to pixel values
            x_center *= w_img
            y_center *= h_img
            w *= w_img
            h *= h_img

            x_min = x_center - w / 2
            y_min = y_center - h / 2

            coco_results.append({
                "image_id": image_id,
                "category_id": int(cls_id) + 1 if class_names is None else class_names[int(cls_id)],
                "bbox": [float(x_min), float(y_min), float(w), float(h)],
                "score": float(score),
                "id": ann_id
            })
            ann_id += 1

    return coco_results

# 1. Build COCO ground-truth JSON
images_path = get_path_in_storage("logistics")
coco_gt = convert_gts_to_coco_format(gts, images_path)

# Save to JSON for pycocotools
gt_json = "ground_truths.json"
with open(gt_json, "w") as f:
    json.dump(coco_gt, f)

# Build image_id_map from COCO GT to keep preds aligned
image_id_map = {img["file_name"][:-4]: img["id"] for img in coco_gt["images"]}

# 2. Sweep over thresholds
results = []
thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
conf_threshold = 0.5

for nms_thresh in thresholds:
    preds = apply_nms(outputs_2,
                      conf_threshold=conf_threshold,
                      nms_threshold=nms_thresh)

    # Convert preds to COCO format with consistent ids
    coco_preds = convert_to_coco_format(preds,
                                        images_path=images_path,
                                        image_id_map=image_id_map)

    # Compute mAP using pycocotools
    map_score = compute_map(coco_preds, gt_json)

    print(f"Threshold={nms_thresh:.2f}, mAP={map_score:.4f}")
    results.append((nms_thresh, map_score))

# 3. Pick best threshold
best_thresh, best_map = max(results, key=lambda x: x[1])
print(f"\nBest threshold = {best_thresh} with mAP = {best_map:.4f}")


# In[ ]:


results


# In[ ]:




