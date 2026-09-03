import os

WNID_TO_NAME = {
    "n01440764": "tench",
    "n02102040": "English springer",
    "n02979186": "cassette player",
    "n03000684": "chain saw",
    "n03028079": "church",
    "n03394916": "French horn",
    "n03417042": "garbage truck",
    "n03425413": "gas pump",
    "n03445777": "golf ball",
    "n03888257": "parachute",
}


def build_wnid_to_label_index(labels_path):
    with open(labels_path) as f:
        labels = [line.strip() for line in f]

    label_lookup = {name.strip().lower(): idx for idx, name in enumerate(labels)}
    wnid_to_index = {}
    unmatched = []

    for wnid, name in WNID_TO_NAME.items():
        key = name.strip().lower()
        if key in label_lookup:
            wnid_to_index[wnid] = label_lookup[key]
        else:
            unmatched.append((wnid, name))

    if unmatched:
        print("Warning: unmatched Imagenette classes:")
        for wnid, name in unmatched:
            print(f"  {wnid}: '{name}'")

    return wnid_to_index, labels


def true_label_for_image(image_path, wnid_to_label_index, labels):
    """
    Given an image path like '.../val/n01440764/n01440764_8622.JPEG',
    extract the WordNet ID from the parent folder name and look up
    the true label text.
    """
    wnid = os.path.basename(os.path.dirname(image_path))
    if wnid not in wnid_to_label_index:
        return None
    return labels[wnid_to_label_index[wnid]]