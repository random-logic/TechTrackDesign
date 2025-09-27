# %%
from typing import Dict, List, Tuple, Any, Callable
import h5py
import os
import h5py
import cv2
from PIL import Image
from cv2.dnn import Net
import numpy as np

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

def get_output_helper(net: Net, img_dims: Tuple[int, int]) -> List[Tuple[int, int, int, int, float, int]]:
    """
    Output - Each detection is going to have x, y, width, height, confidence score, class id
    where x,y are the top left corner of the detection in absolute pixels
    and width, height are in absolute pixels
    """
    outputs = net.forward(get_output_layers(net))
    
    res = []
    for feature_maps in outputs:
        for detection in feature_maps:
            box = detection[:4]
            xywh = cxcywh_norm_to_xywh(*box, img_dims)

            objectness_score = detection[4]
            class_scores = detection[5:]
            class_id = np.argmax(class_scores)
            confidence_score = objectness_score * class_scores[class_id]

            res.append((*xywh, confidence_score, class_id))

    return res

def get_outputs(model_num: int, gts: Dict[str, Any], img_dims = (640, 640)) -> Dict[str, List[Tuple[int, int, int, int, float, float, int]]]:
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

def get_nms(pred: Dict[str, List[Tuple[int, int, int, int, float, float, int]]],
            score_threshold=0.5,
            nms_threshold=0.6) -> Dict[str, List[Tuple[int, int, int, int, float, float, int]]]:
    res = {}
    for key, dets in pred.items():
        bboxes = [list(det[:4]) for det in dets]
        scores = [det[4] * det[5] for det in dets]

        indices = cv2.dnn.NMSBoxes(bboxes, scores, score_threshold, nms_threshold)

        # normalize indices into a flat list
        if indices is None or len(indices) == 0:
            res[key] = []
        else:
            indices = np.array(indices).flatten().tolist()
            res[key] = [dets[i] for i in indices]

    return res

# %%
# Detect all the negative examples incorrectly classified as positive
out2 = load_detections_from_h5(2)
gts = load_gts_from_h5(2)

for fname, dets in out2.items():
    img_gts = gts[fname]

