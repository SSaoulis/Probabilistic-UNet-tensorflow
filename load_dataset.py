import os
import cv2
import tensorflow as tf
import numpy as np

from concurrent.futures import ThreadPoolExecutor
from itertools import chain

NUM_MASKS = 4


def load_nodule(nodule_fp):

    nodule_set = []
    num_slices = len(os.listdir(os.path.join(nodule_fp, "images")))
    for slice in range(num_slices):
        slice_image_path = os.path.join(nodule_fp, "images", f"slice-{slice}.png")
        slice_image = np.expand_dims(
            cv2.imread(slice_image_path, cv2.IMREAD_GRAYSCALE), axis=-1
        )

        for i in range(NUM_MASKS):
            mask_fp = (
                str(os.path.join(nodule_fp, f"mask-{i}", f"slice-{slice}")) + ".png"
            )
            mask = np.expand_dims(cv2.imread(mask_fp, cv2.IMREAD_GRAYSCALE), axis=-1)
            nodule_set.append((slice_image, mask))

    return nodule_set


def load_case(case_fp):

    nodules = []

    for nodule_fp in os.listdir(case_fp):
        nodule_fp = os.path.join(case_fp, nodule_fp)
        if os.path.isdir(nodule_fp):
            nodule_set = load_nodule(nodule_fp)
            nodules += nodule_set

    return nodules


def make_tf_dataset(data_list, batch_size=32, shuffle=True):
    # Separate the tuples into two lists: data and labels
    data, labels = zip(*data_list)

    def normalise_image(image, label):
        image = tf.cast(image, tf.float32) / 255.0
        label = tf.cast(label, tf.float32) / 255.0
        return image, label

    # Create the dataset
    dataset = tf.data.Dataset.from_tensor_slices((list(data), list(labels)))

    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(data_list))

    dataset = (
        dataset.batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
        .map(normalise_image, num_parallel_calls=tf.data.AUTOTUNE)
    )
    return dataset


def train_val_test_split(
    dataset_path, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, total_dataset_size=100
):
    assert (
        abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    ), "Ratios must sum to 1."

    case_fps = os.listdir(dataset_path)
    case_fps = sorted(case_fps)
    # random.shuffle(case_fps)

    # Trim the list if a total size is specified
    if total_dataset_size:
        case_fps = case_fps[:total_dataset_size]

    num_total = len(case_fps)
    num_train = int(train_ratio * num_total)
    num_val = int(val_ratio * num_total)

    train_case_fps = case_fps[:num_train]
    val_case_fps = case_fps[num_train : num_train + num_val]
    test_case_fps = case_fps[num_train + num_val :]

    def load_cases_parallel(file_paths, dataset_path, max_workers=12):
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = executor.map(
                lambda fp: load_case(os.path.join(dataset_path, fp)), file_paths
            )
            return list(chain.from_iterable(results))

    train_data_raw = load_cases_parallel(train_case_fps, dataset_path)
    val_data_raw = load_cases_parallel(val_case_fps, dataset_path)
    test_data_raw = load_cases_parallel(test_case_fps, dataset_path)

    train_ds = make_tf_dataset(train_data_raw, batch_size=32)
    val_ds = make_tf_dataset(val_data_raw, batch_size=32, shuffle=False)
    test_ds = make_tf_dataset(test_data_raw, batch_size=32, shuffle=False)
    return train_ds, val_ds, test_ds
