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
    path = get_path_in_storage(f"outputs_{model_num}{'_nms' if nms else ''}.h5")
    with h5py.File(path, "w") as f:
        for key, tup in outputs.items():
            grp = f.create_group(str(key))
            for i, arr in enumerate(tup):
                grp.create_dataset(
                    f"array_{i}", data=arr, compression="gzip", compression_opts=1
                )

def load_outputs(model_num: int, nms: bool = False) -> Dict[str, Tuple[np.ndarray, ...]]:
    """Load a dict of tuples of arrays from an HDF5 file."""
    path = get_path_in_storage(f"outputs_{model_num}{'_nms' if nms else ''}.h5")
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

# ADD YOUR IMPLEMENTATION HERE
# DO NOT CHANGE THE ABOVE CODE

from torchmetrics.detection.mean_ap import MeanAveragePrecision
import torch

# get ground truths
gts = get_ground_truths()

def convert_for_torchmetrics(preds: Dict[str, Tuple[np.ndarray, ...]], gts: Dict[str, np.ndarray]):
    torch_preds = []
    torch_gts = []
    for fname, dets in preds.items():
        dets = dets[0]  # shape (N, 85)
        if dets.shape[0] > 0:
            boxes = []
            scores = []
            labels = []
            for det in dets:
                x, y, w, h = det[0:4]
                # convert from (x,y,w,h) to (xmin,ymin,xmax,ymax)
                boxes.append([x, y, x + w, y + h])
                scores.append(float(torch.max(torch.tensor(det[5:]))))
                labels.append(int(torch.argmax(torch.tensor(det[5:]))))
            torch_preds.append({
                "boxes": torch.tensor(boxes, dtype=torch.float32),
                "scores": torch.tensor(scores, dtype=torch.float32),
                "labels": torch.tensor(labels, dtype=torch.int64),
            })
        else:
            torch_preds.append({
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "scores": torch.zeros((0,), dtype=torch.float32),
                "labels": torch.zeros((0,), dtype=torch.int64),
            })

        # ground truths
        gt_arr = gts[fname]
        boxes = []
        labels = []
        for gt in gt_arr:
            cls, x, y, w, h = gt
            boxes.append([x, y, x + w, y + h])
            labels.append(int(cls))
        torch_gts.append({
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
        })
    return torch_preds, torch_gts

nms_thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
for nms_th in [0.7]:
    preds_nms = apply_nms(outputs_2, nms_threshold=nms_th)
    torch_preds, torch_gts = convert_for_torchmetrics(preds_nms, gts)

    metric = MeanAveragePrecision()
    metric.update(torch_preds, torch_gts)
    result = metric.compute()
    print(f"NMS threshold {nms_th}: {result}")

# %%
