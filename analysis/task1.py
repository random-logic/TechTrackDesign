# %%
# Functions

from typing import Dict, List, Tuple
import h5py
import os
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

def get_model_paths(model_num: int):
    path = get_path_in_parent(f"yolo_model_{model_num}")
    return (
        os.path.join(path, f"yolov4-tiny-logistics_size_416_{model_num}.weights"),
        os.path.join(path, f"yolov4-tiny-logistics_size_416_{model_num}.cfg")
    )

def get_output_layers(net: Net) -> List[int]:
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

def get_output(net: Net, img_dims: Tuple[int, int]) -> List[Tuple[int, int, int, int, float, int]]:
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

def get_outputs(model_num: int, img_dims = (640, 640)) -> Dict[str, List[Tuple[int, int, int, int, float, int]]]:
    """
    Input - the model number
    Output - for each filename (excluding extension) in the dictionary, a list of detections
    Each detection is going to have x, y, width, height, confidence score, class id
    where x,y are the top left corner of the detection in absolute pixels
    and width, height are in absolute pixels
    """

    net = cv2.dnn.readNet(*get_model_paths(model_num))
        
    res = {}
    for filename in list_logistics_dir():
        if not filename.lower().endswith(".jpg"):
            continue
        
        image = np.array(Image.open(get_logistics_path(filename)))
        blob = cv2.dnn.blobFromImage(image,
                                     scalefactor = 1 / 255.,
                                     size=(416, 416),
                                     mean=(0, 0, 0),
                                     swapRB=True,
                                     crop=False)
        
        net.setInput(blob)
        res[filename[:-4]] = get_output(net, img_dims)

    return res

def save_detections_to_h5(
    detections: Dict[str, List[Tuple[int, int, int, int, float, int]]],
    model_num: int
) -> None:
    """
    Save detections as an HDF5 file.

    detections: Dictionary mapping string keys to lists of tuples (x, y, w, h, confidence, class_id)
    model_num: int - the model number to use
    """
    with h5py.File(get_path_in_storage(f"out{model_num}.h5"), "w", libver="latest") as f:
        for key, det_list in detections.items():
            arr = np.empty((len(det_list), 6), dtype=np.float32)
            for i, (x, y, w, h, confidence, class_id) in enumerate(det_list):
                arr[i, 0] = int(x)
                arr[i, 1] = int(y)
                arr[i, 2] = int(w)
                arr[i, 3] = int(h)
                arr[i, 4] = np.float32(confidence)
                arr[i, 5] = int(class_id)
            f.create_dataset(key, data=arr, compression="gzip", compression_opts=1)

def load_detections_from_h5(
    model_num: int
) -> Dict[str, List[Tuple[int, int, int, int, float, int]]]:
    """
    Load detections from an HDF5 file.
    model_num: int - the model num to use

    Returns a dictionary mapping string keys to lists of tuples (x, y, w, h, confidence, class_id)
    """
    detections = {}
    with h5py.File(get_path_in_storage(f"out{model_num}.h5"), "r") as f:
        for key in f.keys():
            arr = f[key][()]
            detections[key] = [tuple(row) for row in arr]
    return detections

def get_gts(img_dims = (640, 640)):
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
                xywh = cxcywh_norm_to_xywh(cxcywh_norm, img_dims)
                
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
# Get outputs with model 1
out1 = get_outputs(1)

# %%
# Save outputs of model 1
save_detections_to_h5(out1, 1)

# %%
# Get outputs with model 2
out2 = get_outputs(2)

# %%
# Save outputs of model 2
save_detections_to_h5(out2, 2)

# %%
# Load outputs
out1 = load_detections_from_h5(1)
out2 = load_detections_from_h5(2)

# %%
# Get nms
out1_nms = get_nms(out1)
out2_nms = get_nms(out2)

# %%


# %%
