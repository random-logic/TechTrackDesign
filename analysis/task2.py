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

# %%
import random
from imblearn.over_sampling import RandomOverSampler

# Per-class mAPs
per_class_mAP = np.array([
    0.1780,  # barcode
    0.2999,  # car
    0.3629,  # cardboard box
    0.0000,  # fire
    0.2241,  # forklift
    0.0320,  # freight container
    0.3128,  # gloves
    0.0973,  # helmet
    0.0557,  # ladder
    0.1453,  # license plate
    0.1328,  # person
    0.4097,  # qr code
    0.0435,  # road sign
    0.0792,  # safety vest
    0.0469,  # smoke
    0.1972,  # traffic cone
    0.2329,  # traffic light
    0.3380,  # truck
    0.4879,  # van
    0.0261,  # wood pallet
])

def stratified_balanced_sample(
    gts: Dict[str, List[Tuple[int, int, int, int, int]]],
    total_samples: int = 15000,
    underperf_threshold: float = 0.15,
    underperf_weight_factor: float = 2.0,
    n_classes: int = 20
) -> Dict[str, List[Tuple[int, int, int, int, int]]]:
    """
    Perform stratified sampling with balance adjustment.
    
    Args:
        gts: dict mapping image_id -> list of (class_id, x, y, w, h)
        class_id_to_name: mapping of class_id -> readable name
        total_samples: how many total samples to return
        underperf_threshold: mAP below which we consider a class underperforming
        oversample_factor: how much to boost underperforming classes
    
    Returns:
        dict with sampled images and annotations
    """
    # Convert format
    gts_by_class_id: List[List[Tuple[int, int, int, int, str]]] = [[] for _ in range(n_classes)]
    for fname, objs in gts.items():
        for obj in objs:
            class_id = obj[0]
            gts_by_class_id[class_id].append((*obj[1:], fname))

    # calculate the ratio of samples needed for each class
    ratio = np.array([len(gts_by_class_id[i]) for i in range(n_classes)], dtype=float)
    ratio = np.where(per_class_mAP < underperf_threshold, ratio * underperf_weight_factor, ratio)
    ratio /= ratio.sum()
    samples_per_class = ratio * total_samples
    print(samples_per_class)

    # do sampling
    sampled_gts_by_class_id: List[List[Tuple[int, int, int, int, str]]] = [[] for _ in range(n_classes)]
    for i, class_to_sample in enumerate(gts_by_class_id):
        num_of_samples_desired = int(samples_per_class[i])
        num_to_sample = min(len(class_to_sample), num_of_samples_desired)
        sampled_gts_by_class_id[i] = random.sample(class_to_sample, num_to_sample)

    # convert back to original format
    res: Dict[str, List[Tuple[int, int, int, int, int]]] = {}
    for i, objs_at_class_id in enumerate(sampled_gts_by_class_id):
        for obj in objs_at_class_id:
            key = obj[-1]
            if res.get(key, False):
                res[key].append((i, *obj[:-1]))
            else:
                res[key] = [(i, *obj[:-1])]

    return res

# %%
out2 = load_detections_from_h5(2)
gts = get_gts()  # your ground truths
class_id_to_name = {
    0: "barcode",
    1: "car",
    2: "cardboard box",
    3: "fire",
    4: "forklift",
    5: "freight container",
    6: "gloves",
    7: "helmet",
    8: "ladder",
    9: "license plate",
    10: "person",
    11: "qr code",
    12: "road sign",
    13: "safety vest",
    14: "smoke",
    15: "traffic cone",
    16: "traffic light",
    17: "truck",
    18: "van",
    19: "wood pallet",
}
balanced_subset = stratified_balanced_sample(gts)
print(f"Selected {len(balanced_subset)} images")

# %%
from collections import Counter

class_counts = Counter()
for anns in balanced_subset.values():
    for ann in anns:
        cls_id = ann[0]
        class_counts[cls_id] += 1

total_instances = sum(class_counts.values())
print("Class distribution in sampled dataset:")
for cls_id, count in class_counts.items():
    ratio = count / total_instances
    print(f"{class_id_to_name[cls_id]}: {count} ({ratio:.2%})")

# Check for duplicate image IDs
all_ids = list(balanced_subset.keys())
if len(all_ids) == len(set(all_ids)):
    print("✅ No duplicate image IDs found.")
else:
    print("⚠️ Duplicate image IDs detected.")

# %%
# Total class count for each class in the unsampled version
unsampled_class_counts = Counter()
for anns in gts.values():
    for ann in anns:
        cls_id = ann[0]
        unsampled_class_counts[cls_id] += 1

total_unsampled = sum(unsampled_class_counts.values())
print("Class distribution in unsampled dataset:")
for cls_id, count in unsampled_class_counts.items():
    ratio = count / total_unsampled
    print(f"{class_id_to_name[cls_id]}: {count} ({ratio:.2%})")

# %%
def save_detections_to_h5(
    detections: Dict[str, List[Tuple[int, int, int, int, float, int]]],
    model_num: int
) -> None:
    """
    Save detections as an HDF5 file.

    detections: Dictionary mapping string keys to lists of tuples (x, y, w, h, confidence, class_id)
    model_num: int - the model number to use
    """
    with h5py.File(get_path_in_storage(f"out{model_num}_sampled.h5"), "w", libver="latest") as f:
        for key, det_list in detections.items():
            data_list = []
            for (x, y, w, h, confidence, class_id) in det_list:
                data_list.append([x, y, w, h, round(confidence, 8), class_id])
            f.create_dataset(key, data=data_list, dtype="float32", compression="gzip", compression_opts=1)

# %%
# Filter gts to only include images that are in balanced_subset
# Filter out2 to only include images that are in balanced_subset
filtered_out2 = {k: v for k, v in out2.items() if k in balanced_subset}
filtered_gts = {k: v for k, v in gts.items() if k in balanced_subset}

# Save
save_detections_to_h5(filtered_out2, 2)

# %%
def save_gts_to_h5(
    gts: Dict[str, List[Tuple[int, int, int, int, int]]],
    model_num: int
) -> None:
    """
    Save ground truths as an HDF5 file.

    gts: Dictionary mapping string keys to lists of tuples (class_id, x, y, w, h)
    model_num: int - the model number to use
    """
    with h5py.File(get_path_in_storage(f"gts{model_num}_sampled.h5"), "w", libver="latest") as f:
        for key, gt_list in gts.items():
            data_list = []
            for (class_id, x, y, w, h) in gt_list:
                data_list.append([class_id, x, y, w, h])
            f.create_dataset(key, data=data_list, dtype="int32", compression="gzip", compression_opts=1)

save_gts_to_h5(filtered_gts, 2)

# %%
def load_gts_from_h5(
    model_num: int
) -> Dict[str, List[Tuple[int, int, int, int, int]]]:
    """
    Load ground truths from an HDF5 file.

    Returns a dictionary mapping string keys to lists of tuples (class_id, x, y, w, h)
    """
    gts = {}
    with h5py.File(get_path_in_storage(f"gts{model_num}_sampled.h5"), "r") as f:
        for key in f.keys():
            arr = f[key][()]
            gts[key] = [(int(row[0]), int(row[1]), int(row[2]), int(row[3]), int(row[4])) for row in arr]
    return gts

# %%
# Example usage:
sampled_gts = load_gts_from_h5(2)
print(f"Loaded {len(sampled_gts)} sampled ground truth images")
