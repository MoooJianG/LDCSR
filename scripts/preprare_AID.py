import os
import pickle
from natsort import natsorted
import numpy as np

np.random.seed(1)


data_root = "load/AID"
train_scale = 0.8
val_scale = 0.1
test_scale = 0.1

sub_cats = os.listdir(data_root)

train_split, val_split, test_split = [], [], []

for cat in sub_cats:
    cat_path = os.path.join(data_root, cat)
    if os.path.isdir(cat_path):
        images = os.listdir(cat_path)
        np.random.shuffle(images)
        _train_split = images[: int(len(images) * train_scale)]
        _val_split = images[
            int(len(images) * train_scale) : int(
                len(images) * (train_scale + val_scale)
            )
        ]
        _test_split = images[int(len(images) * (train_scale + val_scale)) :]
        train_split.extend([os.path.join(cat, img) for img in natsorted(_train_split)])
        val_split.extend([os.path.join(cat, img) for img in natsorted(_val_split)])
        test_split.extend([os.path.join(cat, img) for img in natsorted(_test_split)])

train_val_test_split = {"train": train_split, "val": val_split, "test": test_split}
pickle.dump(train_val_test_split, open("load/AID_split.pkl", "wb"))
