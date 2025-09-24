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
            data_list = []
            for (x, y, w, h, confidence, class_id) in det_list:
                data_list.append([x, y, w, h, round(confidence, 8), class_id])
            f.create_dataset(key, data=data_list, dtype="float32", compression="gzip", compression_opts=1)

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
# Debug nms
print(f"Model 1: Total predictions before NMS = {sum(len(v) for v in out1.values())}")
print(f"Model 1: Total predictions after NMS = {sum(len(v) for v in out1_nms.values())}")
print(f"Model 2: Total predictions before NMS = {sum(len(v) for v in out2.values())}")
print(f"Model 2: Total predictions after NMS = {sum(len(v) for v in out2_nms.values())}")

# %%
# Visualize predictions for model 1 and model 2
def visualize_predictions(out_nms: Dict[str, List[Tuple[int, int, int, int, float, int]]], model_num: int):
    """
    Draw boxes from out_nms and write annotated images to storage/predictions{model_num}/.
    This function is defensive: it casts types, clamps to image bounds, and skips malformed items.
    """
    output_dir = get_path_in_storage(f"predictions{model_num}")
    os.makedirs(output_dir, exist_ok=True)

    for filename, detections in out_nms.items():
        img_path = get_logistics_path(filename + ".jpg")
        image = cv2.imread(img_path)
        if image is None:
            print(f"visualize_predictions: warning - image not found or unreadable: {img_path}")
            continue

        img_h, img_w = image.shape[:2]

        for det in detections:
            # expect det to be (x, y, w, h, confidence, class_id)
            if not (isinstance(det, (list, tuple)) and len(det) == 6):
                print(f"visualize_predictions: skipping malformed detection for {filename}: {det}")
                continue

            x_raw, y_raw, w_raw, h_raw, conf_raw, cls_raw = det

            # robust casting
            try:
                x = int(x_raw)
                y = int(y_raw)
                box_w = int(w_raw)
                box_h = int(h_raw)
                confidence = float(conf_raw)
                class_id = int(cls_raw)
            except Exception:
                # fallback: try casting via float then int
                try:
                    x = int(float(x_raw))
                    y = int(float(y_raw))
                    box_w = int(float(w_raw))
                    box_h = int(float(h_raw))
                    confidence = float(conf_raw)
                    class_id = int(float(cls_raw))
                except Exception as e:
                    print(f"visualize_predictions: skipping detection with bad types for {filename}: {det} -> {e}")
                    continue

            # normalize and clamp coordinates to image bounds
            if box_w < 0:
                box_w = 0
            if box_h < 0:
                box_h = 0

            x = max(0, min(x, img_w - 1))
            y = max(0, min(y, img_h - 1))
            x2 = max(0, min(x + box_w, img_w - 1))
            y2 = max(0, min(y + box_h, img_h - 1))

            top_left = (x, y)
            bottom_right = (x2, y2)

            # draw rectangle (wrapped in try to catch any odd cv2 type issues)
            try:
                cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)
            except Exception as e:
                print(f"visualize_predictions: cv2.rectangle failed for {filename} det {det}: {e}")
                continue

            # draw label above box if possible, otherwise below
            label = f"{class_id}: {confidence:.2f}"
            text_y = y - 10 if y - 10 > 10 else y + 20
            cv2.putText(image, label, (x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        save_path = os.path.join(output_dir, filename + ".jpg")
        cv2.imwrite(save_path, image)

# calls
visualize_predictions(out1_nms, 1)
visualize_predictions(out2_nms, 2)

# %%
# Get Coco mAP curve
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

gts = get_gts()
gts_coco, pred_coco = get_coco_gts_and_preds(gts, out2_nms)

# %%
gts_coco

# %%
pred_coco

# %%

