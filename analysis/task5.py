# %%
from typing import Dict, List, Tuple, Any, Callable
import h5py
import os
import h5py
import cv2
from PIL import Image
from cv2.dnn import Net
import numpy as np

Det = Tuple[
    int, int, int, int,  # bounding box
    float, # objectness
    List[float] # classes
]

def get_path_in_parent(*args):
    return os.path.abspath(os.path.join(os.getcwd(), '..', *args))

def get_path_in_storage(*args):
    return get_path_in_parent("storage", *args)

def get_logistics_path(*args):
    return get_path_in_storage("logistics", *args)

def list_logistics_dir():
    return os.listdir(get_logistics_path())

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


def get_model_paths(model_num: int):
    path = get_path_in_parent(f"yolo_model_{model_num}")
    return (
        os.path.join(path, f"yolov4-tiny-logistics_size_416_{model_num}.weights"),
        os.path.join(path, f"yolov4-tiny-logistics_size_416_{model_num}.cfg")
    )

def get_output_layers(net: Net) -> List[str]:
    """
    Returns the indices of the output layers in a cv2.dnn net
    """
    layer_names = net.getLayerNames()
    return [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

def cxcywh_norm_to_xywh(cx_norm: float, cy_norm: float, w_norm: float, h_norm: float, img_dims: Tuple[int, int]) -> Tuple[int, int, int, int]:
    x_norm, y_norm = cx_norm - w_norm / 2, cy_norm - h_norm / 2
    x, w = np.array([x_norm, w_norm]) * img_dims[0]
    y, h = np.array([y_norm, h_norm]) * img_dims[1]
    x, y, w, h = map(int, np.array([x, y, w, h]) + 0.5)
    return x, y, w, h

def get_output_helper(net: Net, img_dims: Tuple[int, int]) -> List[Det]:
    """
    Output - Each detection is going to have x, y, width, height, confidence score, class id
    where x,y are the top left corner of the detection in absolute pixels
    and width, height are in absolute pixels
    """
    outputs = net.forward(get_output_layers(net))
    
    res = []
    for feature_maps in outputs:
        for detection in feature_maps:
            xywh = cxcywh_norm_to_xywh(*detection[:4], img_dims)
            res.append((*xywh, detection[4], detection[5:]))

    return res

def get_outputs(model_num: int, gts: Dict[str, Any], img_dims = (640, 640)) -> Dict[str, List[Det]]:
    """
    Input - the model number, the ground truths
    Output - for each filename (excluding extension) in the dictionary, a list of detections
    Each detection is going to have x, y, width, height, confidence score, class id
    where x,y are the top left corner of the detection in absolute pixels
    and width, height are in absolute pixels
    """

    net = cv2.dnn.readNet(*get_model_paths(model_num))
        
    res = {}
    for filename in gts:
        image = np.array(Image.open(get_logistics_path(f"{filename}.jpg")))
        blob = cv2.dnn.blobFromImage(image,
                                     scalefactor = 1 / 255.,
                                     size=(416, 416),
                                     mean=(0, 0, 0),
                                     swapRB=True,
                                     crop=False)
        
        net.setInput(blob)
        res[filename] = get_output_helper(net, img_dims)

    return res

def get_nms(pred: Dict[str, List[Det]],
            score_threshold=0.5,
            nms_threshold=0.6) -> Dict[str, List[Det]]:
    res = {}
    for key, dets in pred.items():
        bboxes = [list(det[:4]) for det in dets]
        scores = [det[4] * det[5][np.argmax(det[5])] for det in dets]

        indices = cv2.dnn.NMSBoxes(bboxes, scores, score_threshold, nms_threshold)

        # normalize indices into a flat list
        if indices is None or len(indices) == 0:
            res[key] = []
        else:
            indices = np.array(indices).flatten().tolist()
            res[key] = [dets[i] for i in indices]

    return res

def save_outputs_to_h5(
    outputs: Dict[str, List[Det]],
    model_num: int
) -> None:
    """
    Save detections (outputs) as an HDF5 file.
    Integers stored as int32, floats stored as float32.
    """
    path = get_path_in_storage(f"outputs{model_num}.h5")
    with h5py.File(path, "w") as f:
        for key, dets in outputs.items():
            # Flatten detections into arrays of fixed type
            max_classes = max((len(det[5]) for det in dets), default=0)
            arr = []
            for det in dets:
                x, y, w, h, obj, classes = det
                cls_arr = np.array(classes, dtype=np.float32)
                # pad classes to max length for consistency
                if len(cls_arr) < max_classes:
                    cls_arr = np.pad(cls_arr, (0, max_classes - len(cls_arr)), "constant")
                arr.append([x, y, w, h, obj, *cls_arr])
            if arr:
                dset = np.array(arr, dtype=np.float32)
            else:
                dset = np.empty((0,), dtype=np.float32)
            f.create_dataset(key, data=dset)

def load_outputs_from_h5(
    model_num: int
) -> Dict[str, List[Det]]:
    """
    Load detections (outputs) from an HDF5 file.
    Returns dict[str, List[Det]].
    Ensures ints are cast to python int, floats cast to python float.
    """
    path = get_path_in_storage(f"outputs{model_num}.h5")
    res: Dict[str, List[Det]] = {}
    with h5py.File(path, "r") as f:
        for key in f.keys():
            data = f[key][()]
            dets: List[Det] = []
            for row in data:
                x, y, w, h = map(int, row[:4])
                obj = float(row[4])
                classes = [float(c) for c in row[5:]]
                dets.append((x, y, w, h, obj, classes))
            res[key] = dets
    return res

# %%
# Detect all the negative examples incorrectly classified as positive
gts = load_gts_from_h5(2)
out2 = get_nms(get_outputs(2, gts))

save_outputs_to_h5(out2, 2)

# %%
# Load file
out2 = load_outputs_from_h5(2)

# %%
# IOU
def iou(box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        xa, ya = max(x1, x2), max(y1, y2)
        xb, yb = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
        inter = max(0, xb - xa) * max(0, yb - ya)
        union = w1 * h1 + w2 * h2 - inter
        return inter / union if union > 0 else 0.0

# %%
# Map dets to gts
def get_gt_idx_with_highest_iou(
    det: Det,
    img_gts: List[Tuple[int, int, int, int, int]]
) -> Tuple[int, float]:
    matched_gt = -1
    highest_iou = 0.0

    for i, gt in enumerate(img_gts):
        gt_det_iou = iou(gt[1:], det[:4])
        if gt_det_iou > highest_iou:
            matched_gt = i
            highest_iou = gt_det_iou

    return matched_gt, highest_iou


def map_dets_to_gts(
    dets: List[Det],
    img_gts: List[Tuple[int, int, int, int, int]]
) -> List[Tuple[Det, Tuple[int, int, int, int, int] | None]]:
    # map one ground truth to one detection
    gt_to_det_idx: int = [-1] * len(img_gts)
    gt_to_det_iou: float = [0] * len(img_gts)
    for det_idx, det in enumerate(dets):
        matched_gt_idx, matched_gt_iou = get_gt_idx_with_highest_iou(det, img_gts)
        
        # do not keep anything that is below our iou threshold
        if matched_gt_iou < 0.5:
            continue
        
        # if conflicting det in gt, keep the one with highest iou
        gt_to_det_iou[matched_gt_idx], gt_to_det_idx[matched_gt_idx] = max(
            (gt_to_det_iou[matched_gt_idx], gt_to_det_idx[matched_gt_idx]),
            (matched_gt_iou, det_idx)
        )
    
    del gt_to_det_iou

    # Map detections idx to ground truths idx
    det_to_gt_idx: int = [-1] * len(dets)
    for gt_idx, det_idx in enumerate(gt_to_det_idx):
        det_to_gt_idx[det_idx] = gt_idx

    del gt_to_det_idx
    
    # Map the actual detections to actual ground truths
    res = []
    for det_idx, det in enumerate(dets):
        gt_idx = det_to_gt_idx[det_idx]
        if gt_idx >= 0:
            res.append((det, img_gts[gt_idx]))
        else:
            res.append((det, None))

    return res

# %%
# Calculate loss
def L_bb(
    det: Det,
    gt: Tuple[int, int, int, int, int]
) -> float:
    x, y, w, h = det[:4]
    xh, yh, wh, hh = gt[1:]
    return (x - xh) ** 2 + (y - yh) ** 2 + (w - wh) ** 2 + (h - hh) ** 2

# TODO - other loss functions

def get_loss(
    dets: List[Det],
    img_gts: List[Tuple[int, int, int, int, int]],
    lambda_bb: float,
    lambda_obj: float,
    lambda_cls: float,
    lambda_no_obj: float
) -> float:
    res = 0.0
    for det, gt in map_dets_to_gts(dets, img_gts):
        if gt is None:
            res += lambda_no_obj
            continue
        # TODO

"""
L_total = 0
N = number of candidate detections
for prediction in predictions:
    if iou(ground_truth, prediction) > iou_threshold :
        L_total += λ_bb L_bb + λ_obj L_obj + λ_cls L_cls
    else:
        L_total += λ_no_obj L_obj
return L_total / N
"""

# %%
for fname, dets in out2.items():
    img_gts = gts[fname]
    
