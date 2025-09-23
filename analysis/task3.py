# %%


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

def get_ground_truths() -> Dict[str, np.ndarray]:
    # returns a dictionary of { filename without extension : [(id, x, y, width, height), ...] }
    # coords are normalized
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

# outputs are in format of
# { filename without extension: tuple of np.ndarrays, same format as cv2.dnn net.forward }
outputs_2: Dict[str, Tuple[np.ndarray, ...]] = load_outputs(2)

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


# %%

# %%

import json
import tempfile

def convert_to_coco_format(preds: Dict[str, Tuple[np.ndarray, ...]],
                           image_id_map: Dict[str, int]) -> list:
    """Convert predictions to COCO format."""
    coco_results = []
    for fname, layers in preds.items():
        image_id = image_id_map[fname]
        for out in layers:
            for det in out:
                x, y, w, h = det[0:4]
                obj_conf = det[4]
                class_id = int(np.argmax(det[5:]))
                score = float(det[5 + class_id])
                coco_results.append({
                    "image_id": image_id,
                    "category_id": class_id,
                    "bbox": [float(x), float(y), float(w), float(h)],
                    "score": score * float(obj_conf)
                })
    return coco_results

def build_coco_gt(gt_dict: Dict[str, np.ndarray]) -> Tuple[COCO, Dict[str, int]]:
    """Build a COCO object for ground truths."""
    images, annotations, categories = [], [], []
    ann_id = 1
    image_id_map = {}
    cat_ids = set()

    for i, (fname, anns) in enumerate(gt_dict.items()):
        image_id = i + 1
        image_id_map[fname] = image_id
        images.append({"id": image_id, "file_name": fname})
        for ann in anns:
            cat_id = int(ann[0])
            cat_ids.add(cat_id)
            x, y, w, h = ann[1:5]
            annotations.append({
                "id": ann_id,
                "image_id": image_id,
                "category_id": cat_id,
                "bbox": [x, y, w, h],
                "area": w * h,
                "iscrowd": 0
            })
            ann_id += 1

    categories = [{"id": cid, "name": str(cid)} for cid in sorted(cat_ids)]

    coco_dict = {
        "info": {"description": "Ground truth dataset"},
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": categories
    }

    # Save to a temporary JSON file and load with COCO
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(coco_dict, f)
        tmp_path = f.name

    coco_gt = COCO(tmp_path)
    return coco_gt, image_id_map

def compute_map(coco_gt: COCO, coco_preds: list) -> float:
    """Compute mean Average Precision (mAP)."""
    coco_dt = coco_gt.loadRes(coco_preds)
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return coco_eval.stats[0]  # AP at IoU=0.50:0.95

if __name__ == "__main__":
    ground_truths = get_ground_truths()
    coco_gt, image_id_map = build_coco_gt(ground_truths)

    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    for nms_thresh in thresholds:
        preds = apply_nms(outputs_2, conf_threshold=0.5, nms_threshold=nms_thresh)
        coco_preds = convert_to_coco_format(preds, image_id_map)
        print(f"\nEvaluating with NMS threshold = {nms_thresh}")
        map_score = compute_map(coco_gt, coco_preds)
        print(f"mAP: {map_score:.4f}")

# %%
