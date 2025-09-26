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

def cxcywh_norm_to_xywh(cx_norm: float, cy_norm: float, w_norm: float, h_norm: float, img_dims: Tuple[int, int]) -> Tuple[int, int, int, int]:
    x_norm, y_norm = cx_norm - w_norm / 2, cy_norm - h_norm / 2
    x, w = np.array([x_norm, w_norm]) * img_dims[0]
    y, h = np.array([y_norm, h_norm]) * img_dims[1]
    x, y, w, h = map(int, np.array([x, y, w, h]) + 0.5)
    return x, y, w, h

def load_detections_from_h5(
    model_num: int
) -> Dict[str, List[Tuple[int, int, int, int, float, int]]]:
    """
    Load detections from an HDF5 file.
    model_num: int - the model num to use

    Returns a dictionary mapping string keys to lists of tuples (x, y, w, h, confidence, class_id) where x, y, w, h are all absolute pixel values
    """
    detections = {}
    with h5py.File(get_path_in_storage(f"out{model_num}.h5"), "r") as f:
        for key in f.keys():
            arr = f[key][()]
            detections[key] = [(int(row[0]), int(row[1]), int(row[2]), int(row[3]), float(row[4]), int(row[5])) for row in arr]
    return detections

def get_gts(img_dims = (640, 640)) -> Dict[str, List[Tuple[int, int, int, int, int]]]:
    """
    Reads YOLO-format ground truth .txt files from the logistics directory.
    Each line in the file contains: class_id cx cy w h (all normalized coordinates).
    Converts normalized coordinates to absolute pixel values using img_dims.
    Returns a dictionary mapping each filename (without extension) to a list of tuples (class_id, x, y, w, h),
    where x, y, w, h are in absolute pixel values.
    """
    res = {}
    for filename in list_logistics_dir():
        if not filename.lower().endswith(".txt"):
            continue
        
        gts = []
        with open(get_logistics_path(filename), "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                
                class_id = int(parts[0])
                cxcywh_norm = map(float, parts[1:])
                xywh = cxcywh_norm_to_xywh(*cxcywh_norm, img_dims)
                
                gts.append((class_id, *xywh))

        res[filename[:-4]] = gts
    return res

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

# %%
# YOUR CODE HERE
out2 = load_detections_from_h5(2)

