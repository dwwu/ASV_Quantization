import os
import pandas as pd
import shutil

iden_split = pd.read_csv("sv_set/voxc1/iden_split.txt",
        header=None,
        delimiter=' ',
        names=['split', 'file'])

# split file names
train_file = iden_split[iden_split.split == 1].file.tolist()
val_file = iden_split[iden_split.split == 2].file.tolist()
test_file = iden_split[iden_split.split == 3].file.tolist()


# fix file names to match to the feature filenames
fix_name_fn = lambda f: "/".join(f.split("/")[:2]) + "-" + f.split("/")[-1].replace("wav", "npy")
train_file = list(map(fix_name_fn, train_file))
val_file = list(map(fix_name_fn, val_file))
test_file = list(map(fix_name_fn, test_file))


# copy and paste into corresponding directories

for f in train_file:
    src = "sv_set/voxc1/fbank64/"+f
    dst = "sv_set/voxc1/fbank64_dev/train/"+f

    # ignore sv_test files
    # if not os.path.isfile(src):
        # continue

    try:
        shutil.copy(src, dst)
    except IOError as e:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy(src, dst)


for f in val_file:
    src = "sv_set/voxc1/fbank64/"+f
    dst = "sv_set/voxc1/fbank64_dev/val/"+f

    # ignore sv_test files
    # if not os.path.isfile(src):
        # continue

    try:
        shutil.copy(src, dst)
    except IOError as e:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy(src, dst)

for f in test_file:
    src = "sv_set/voxc1/fbank64/"+f
    dst = "sv_set/voxc1/fbank64_dev/test/"+f

    # ignore sv_test files
    # if not os.path.isfile(src):
        # continue

    try:
        shutil.copy(src, dst)
    except IOError as e:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy(src, dst)



