# 利用glob库，使用通配符，过滤系统中的隐藏文件
import glob
import os
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    ScaleIntensityRanged,
    Resized,
    RandCropByPosNegLabeld,
    SqueezeDimd,
)

from pathlib import Path

data_dir = "/data/xuzhang/LungTumorSegmentation/data"


def sort_fn(path: str):
    filename = Path(path).name.split(".")[0]
    if filename.startswith("PET"):
        return int(filename[3:])
    else:
        return int(filename[2:])


pet_images = sorted(
    glob.glob(os.path.join(data_dir, "Data_PET", "*.nii.gz")), key=sort_fn)
pet_labels = sorted(
    glob.glob(os.path.join(data_dir, "Label_PET", "*.nii.gz")), key=sort_fn)
ct_images = sorted(
    glob.glob(os.path.join(data_dir, "Data_CT", "*.nii.gz")), key=sort_fn)
ct_labels = sorted(
    glob.glob(os.path.join(data_dir, "Label_CT", "*.nii.gz")), key=sort_fn)
data_dicts = [
    {"pet_image": pet_image, "pet_label": pet_label,
        "ct_image": ct_image, "ct_label": ct_label}
    for pet_image, pet_label, ct_image, ct_label in zip(pet_images, pet_labels, ct_images, ct_labels)
]
train_files, val_files = data_dicts[:-10], data_dicts[-10:]


keys = ['pet_image', 'pet_label', 'ct_image', 'ct_label']
train_transforms = Compose(
    [
        LoadImaged(keys=keys),
        EnsureChannelFirstd(keys=keys),
        Resized(keys=keys, spatial_size=(512, 512, -1)),
        ScaleIntensityRanged(
            keys=['pet_image', 'ct_image'], a_max=255, a_min=0, b_max=1, b_min=0),
        RandCropByPosNegLabeld(
            keys=["pet_image", "pet_label", "ct_image", "ct_label"],
            label_key="pet_label",
            spatial_size=(512, 512, 1),
            pos=1,
            neg=1,
            num_samples=8,
            image_key="pet_image",
            image_threshold=0,),
        SqueezeDimd(keys=["pet_image", "pet_label",
                    "ct_image", "ct_label"], dim=-1),
    ]
)

val_transforms = Compose(
    [
        LoadImaged(keys=keys),
        EnsureChannelFirstd(keys=keys),
        Resized(keys=keys, spatial_size=(512, 512, -1)),
        ScaleIntensityRanged(
            keys=['pet_image', 'ct_image'], a_max=255, a_min=0, b_max=1, b_min=0),
    ]
)


if __name__ == "__main__":
    print(val_files)
