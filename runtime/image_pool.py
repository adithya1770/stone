import os
import random

DEFAULT_DATASET_DIR = "datasets/imagenette2-320/val"
DEFAULT_POOL_SIZE = 150
DEFAULT_SEED = 42


def build_image_pool(dataset_dir=DEFAULT_DATASET_DIR,
                      pool_size=DEFAULT_POOL_SIZE,
                      seed=DEFAULT_SEED):
    """
    Scans the dataset directory for images, picks a fixed, reproducible
    subset of `pool_size` images (same subset, same order, every time
    this is called with the same seed), and returns it as a list of
    file paths.

    Fixed order matters here: every algorithm/experiment run should see
    the exact same sequence of images, so any difference in results is
    due to the algorithm's behaviour, not which images happened to show
    up when.
    """
    all_images = []

    for class_dir in sorted(os.listdir(dataset_dir)):
        class_path = os.path.join(dataset_dir, class_dir)
        if not os.path.isdir(class_path):
            continue
        for fname in sorted(os.listdir(class_path)):
            if fname.lower().endswith((".jpeg", ".jpg", ".png")):
                all_images.append(os.path.join(class_path, fname))

    if not all_images:
        raise FileNotFoundError(
            f"No images found under {dataset_dir}. "
            f"Check the dataset path."
        )

    rng = random.Random(seed)
    rng.shuffle(all_images)

    pool = all_images[:pool_size]

    if len(pool) < pool_size:
        print(f"Warning: only {len(pool)} images available, "
              f"requested pool size was {pool_size}.")

    return pool


class ImagePool:
    """
    Wraps a fixed image pool and hands out images in a repeatable,
    cycling order. Call get(iteration_index) each loop iteration.
    """

    def __init__(self, dataset_dir=DEFAULT_DATASET_DIR,
                 pool_size=DEFAULT_POOL_SIZE,
                 seed=DEFAULT_SEED):
        self.pool = build_image_pool(dataset_dir, pool_size, seed)
        print(f"ImagePool: loaded {len(self.pool)} images "
              f"(seed={seed}, fixed order).")

    def get(self, iteration_index):
        return self.pool[iteration_index % len(self.pool)]

    def __len__(self):
        return len(self.pool)