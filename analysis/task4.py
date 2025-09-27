# %%
from typing import Dict, List, Tuple, Any, Callable
import h5py
import os
import h5py
import cv2
from PIL import Image
from cv2.dnn import Net
import numpy as np
import json
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

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

def get_outputs(model_num: int, gts: Dict[str, Any], img_transform: Callable[[np.ndarray], np.ndarray], img_dims = (640, 640)) -> Dict[str, List[Tuple[int, int, int, int, float, int]]]:
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
        image = img_transform(image)
        blob = cv2.dnn.blobFromImage(image,
                                     scalefactor = 1 / 255.,
                                     size=(416, 416),
                                     mean=(0, 0, 0),
                                     swapRB=True,
                                     crop=False)
        
        net.setInput(blob)
        res[filename] = get_output_helper(net, img_dims)

    return res

def get_nms(pred: Dict[str, List[Tuple[int, int, int, int, float, int]]],
            score_threshold=0.5,
            nms_threshold=0.6) -> Dict[str, List[Tuple[int, int, int, int, float, int]]]:
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

def gaussian_blur(image: np.ndarray) -> np.ndarray:
    """Apply Gaussian blur to the image."""
    return cv2.GaussianBlur(image, (5, 5), 0)

def vertical_flip(image: np.ndarray) -> np.ndarray:
    """Flip the image vertically."""
    return cv2.flip(image, 0)

def adjust_brightness(image: np.ndarray, factor: float = 1.2) -> np.ndarray:
    """Adjust brightness by multiplying pixel values with a factor."""
    image = image.astype(np.float32) * factor
    return np.clip(image, 0, 255).astype(np.uint8)

# Get Coco formats
def get_coco_gts_and_preds(gts: Dict[str, List[Tuple[int, int, int, int, int]]], preds: Dict[str, List[Tuple[int, int, int, int, float, int]]], img_dims = (640, 640)) -> Tuple[Dict, List]:
    imgs = []
    annotations = []
    pred_res = []

    for img_id, (img_name, img_gts) in enumerate(gts.items()):
        imgs.append({
            "id": int(img_id),
            "file_name": f"{img_name}.jpg",
            "width": int(img_dims[0]),
            "height": int(img_dims[1])
        })

        for class_id, x, y, w, h in img_gts:
            annotations.append({
                "id": int(len(annotations)),
                "image_id": int(img_id),
                "category_id": int(class_id),
                "bbox": [int(x), int(y), int(w), int(h)],
                "area": float(w * h),
                "iscrowd": 0
            })

        for x, y, w, h, conf_score, class_id in preds[img_name]:
            pred_res.append({
                "id": int(len(pred_res)),
                "image_id": int(img_id),
                "category_id": int(class_id),
                "bbox": [int(x), int(y), int(w), int(h)],
                "score": float(conf_score)
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

def evaluate(img_filter: Callable[[np.ndarray], np.ndarray]):
    gts = load_gts_from_h5(2)
    out2 = get_nms(get_outputs(2, gts, img_filter))

    gts_coco_2, pred_coco_2 = get_coco_gts_and_preds(gts, out2)

    with open(get_path_in_storage("gts_coco_2t.json"), "w") as f:
        json.dump(gts_coco_2, f)
    with open(get_path_in_storage("pred_coco_2t.json"), "w") as f:
        json.dump(pred_coco_2, f)

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
    coco_gt = COCO(get_path_in_storage("gts_coco_2t.json"))
    coco_dt = coco_gt.loadRes(get_path_in_storage("pred_coco_2t.json"))

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
evaluate(gaussian_blur)

# %%
evaluate(vertical_flip)

# %%
evaluate(adjust_brightness)

# %%
