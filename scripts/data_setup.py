#!/usr/bin/env python3
"""
Functionality for creating lettuce dataset as PyTorch Dataset
and loading thre train & test datasets.

TODO:
    - Apply RGB masks to Fluorescence images in folder
"""
# Import statements
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from PIL import Image
from sklearn.model_selection import train_test_split

# Import supporting modules
import utils


# Define Pytorch Dataset class for lettuce dataset
class LettuceSegDataset(Dataset):
    def __init__(
        self,
        img_dir,
        label_dir,
        is_train,
        fm_dir=None,
        fvfm_dir=None,
        train_frac=0.75,
        transform=None,
        seed=42,
        give_name=False,
    ):
        """Creates a PyTorch Dataset class of the lettuce segmantation dataset.

        Uses sorted() to match filenames of images and masks. Check if filenames
        are constructed in such a way that sorted() will sort the filenames in the same way.
        Otherwise images and masks will be mismatched when loading the data.

        Fm and FvFm images can be included as additional input. The Fm and FvFm are stacked
        with the RGB images as additional channels.

        Args:
            img_dir (str): Filepath of directory containing the images.
            label_dir (str): Filepath of directory  containing the segmentation masks.
            is_train (bool): If true, gives train data. If false, gives test data.
            fm_dir (str, optional): Filepath of directory containing Fm images for channel stacked input.
            fvfm_dir (str, optional): Filepath of directory containing FvFm images for channel stacked input.
            train_frac (float, optional): Fraction of data that is train. Defaults to 0.75.
            transform (albumentations.Compose, optional): Transformations for data aug. Defaults to None.
            seed (int, optional): Seed for reproducible train test split of data. Defaults to 42.
            give_name (bool, optional): If True, dataset also provides image name.
        """
        self.transform = transform
        self.give_name = give_name

        # List all image, mask filenames and optionally Fm and FvFm filenames
        self.img_names = sorted(os.listdir(img_dir))
        mask_names = sorted(os.listdir(label_dir))
        if fm_dir is not None:
            fm_names = sorted(os.listdir(fm_dir))
        if fvfm_dir is not None:
            fvfm_names = sorted(os.listdir(fvfm_dir))

        # Check if there is an incomplete number of masks
        if len(self.img_names) != len(mask_names):
            incomplete_masks = True
            print(
                "[INFO] Numbers of images and masks are inequal, cancel if unintended!"
            )

        # Create lists of filepaths for images and masks
        if incomplete_masks is False:
            img_paths = []
            mask_paths = []
            for img_name, mask_name in zip(self.img_names, mask_names):
                img_path = os.path.join(img_dir, img_name)
                mask_path = os.path.join(label_dir, mask_name)
                img_paths.append(img_path)
                mask_paths.append(mask_path)

        # Create lists of filepaths for images and masks when there are not masks for every image
        else:
            img_paths = []
            mask_paths = []
            mask_names = np.array(mask_names)
            for img_name in self.img_names:
                img_path = os.path.join(img_dir, img_name)
                img_paths.append(img_path)

                # List raw image name in mask paths for missing masks
                raw_name = img_name.split(os.extsep)[0]
                if raw_name.endswith("_bg_mask"):  # for filenames of bg masked images
                    raw_name = raw_name.removesuffix("_bg_mask")

                match_ind = np.flatnonzero(
                    np.core.defchararray.find(mask_names, raw_name) != -1
                )
                if len(match_ind) == 1:
                    mask_path = os.path.join(label_dir, mask_names[match_ind[0]])
                elif len(match_ind) == 0:
                    mask_path = raw_name
                mask_paths.append(mask_path)

        # Also create lists of filepaths for Fm and FvFm if desired
        if fm_dir is not None:
            fm_paths = []
            for fm_name in fm_names:
                fm_path = os.path.join(fm_dir, fm_name)
                fm_paths.append(fm_path)
        if fvfm_dir is not None:
            fvfm_paths = []
            for fvfm_name in fvfm_names:
                fvfm_path = os.path.join(fvfm_dir, fvfm_name)
                fvfm_paths.append(fvfm_path)

        # Split into train and test sets if desired
        if train_frac < 1:
            if (fm_dir is None) and (fvfm_dir is None):
                split = train_test_split(
                    img_paths,
                    mask_paths,
                    train_size=train_frac,
                    random_state=seed,
                )

                # Give train or test data as requested
                if is_train:
                    self.img_paths = split[0]
                    self.mask_paths = split[2]
                else:
                    self.img_paths = split[1]
                    self.mask_paths = split[3]
            elif (fm_dir is not None) and (fvfm_dir is None):
                split = train_test_split(
                    img_paths,
                    fm_paths,
                    mask_paths,
                    train_size=train_frac,
                    random_state=seed,
                )

                # Give train or test data as requested
                if is_train:
                    self.img_paths = split[0]
                    self.fm_paths = split[2]
                    self.mask_paths = split[4]
                else:
                    self.img_paths = split[1]
                    self.fm_paths = split[3]
                    self.mask_paths = split[5]
            elif (fm_dir is None) and (fvfm_dir is not None):
                split = train_test_split(
                    img_paths,
                    fvfm_paths,
                    mask_paths,
                    train_size=train_frac,
                    random_state=seed,
                )

                # Give train or test data as requested
                if is_train:
                    self.img_paths = split[0]
                    self.fvfm_paths = split[2]
                    self.mask_paths = split[4]
                else:
                    self.img_paths = split[1]
                    self.fvfm_paths = split[3]
                    self.mask_paths = split[5]
            elif (fm_dir is not None) and (fvfm_dir is not None):
                split = train_test_split(
                    img_paths,
                    fm_paths,
                    fvfm_paths,
                    mask_paths,
                    train_size=train_frac,
                    random_state=seed,
                )

                # Give train or test data as requested
                if is_train:
                    self.img_paths = split[0]
                    self.fm_paths = split[2]
                    self.fvfm_paths = split[4]
                    self.mask_paths = split[6]
                else:
                    self.img_paths = split[1]
                    self.fm_paths = split[3]
                    self.fvfm_paths = split[5]
                    self.mask_paths = split[7]
        else:
            self.img_paths = img_paths
            self.mask_paths = mask_paths
            if fm_dir is not None:
                self.fm_paths = fm_paths
            if fvfm_dir is not None:
                self.fvfm_paths = fvfm_paths

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        # Retrieve image, should be np.array for albumentations.transforms
        img = np.array(Image.open(self.img_paths[index]))

        # Also retrieve Fm and FvFm images if desired
        if hasattr(self, "fm_paths"):
            fm = np.array(Image.open(self.fm_paths[index]))
            fm = fm / fm.max() * 255  # normalize Fm values as they are large
        if hasattr(self, "fvfm_paths"):
            fvfm = np.array(Image.open(self.fvfm_paths[index]))

        # Retrieve mask, mask could be .json or an image format
        size = img.shape[:2]
        if self.mask_paths[index].endswith(".json"):
            mask = utils.binary_poly2px(self.mask_paths[index], custom_size=size)
        elif self.mask_paths[index].endswith(
            (".png", ".jpg", ".jpeg", ".tif", ".tiff")
        ):
            mask = np.array(Image.open(self.mask_paths[index]))
        else:
            mask = np.zeros(size, dtype=np.int32)  # Create empty mask for missing mask
        if 255.0 in mask:
            mask[mask == 255.0] = 1.0

        # Apply data augmentation transforms to image, mask and optionally Fm and FvFm
        if self.transform is not None:
            grayscales = [mask]
            if hasattr(self, "fm_paths"):
                grayscales.append(fm)
            if hasattr(self, "fvfm_paths"):
                grayscales.append(fvfm)
            augmentations = self.transform(image=img, masks=grayscales)
            img = augmentations["image"]
            if (not hasattr(self, "fm_paths")) and (not hasattr(self, "fvfm_paths")):
                mask = augmentations["masks"]
            elif (hasattr(self, "fm_paths")) and (not hasattr(self, "fvfm_paths")):
                mask, fm = augmentations["masks"]
            elif (not hasattr(self, "fm_paths")) and (hasattr(self, "fvfm_paths")):
                mask, fvfm = augmentations["masks"]
            elif (hasattr(self, "fm_paths")) and (hasattr(self, "fvfm_paths")):
                mask, fm, fvfm = augmentations["masks"]

        # Apply background mask from RGB to Fm and FvFm
        if hasattr(self, "fm_paths"):
            fm = fm * (img.sum(dim=0) > 0)
        if hasattr(self, "fvfm_paths"):
            fvfm = fvfm * (img.sum(dim=0) > 0)

        # Compile resulting images
        if hasattr(self, "fm_paths"):
            img = np.concatenate([img, fm[np.newaxis, :, :]], axis=0)
        if hasattr(self, "fvfm_paths"):
            img = np.concatenate([img, fvfm[np.newaxis, :, :]], axis=0)
        result = (img, mask)

        # Also provide image name if desired
        if self.give_name:
            img_name = self.img_names[index].split(os.extsep)[0]
            result.append(img_name)

        return result


import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

transforms = A.Compose(
    [
        A.Resize(height=480, width=480),
        A.Rotate(limit=1100, border_mode=cv2.BORDER_CONSTANT, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=0.5),
        A.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.05, p=0.5),
        ToTensorV2(),
    ]
)

dataset = LettuceSegDataset(
    img_dir="/lustre/BIF/nobackup/to001/thesis_MBF/data/TrainTest_tipburn/UnetMit-b3_bg_masks_combined",
    label_dir="/lustre/BIF/nobackup/to001/thesis_MBF/data/TrainTest_tipburn/stitched_tb_masks_combined",
    is_train=True,
    fm_dir="/lustre/BIF/nobackup/to001/thesis_MBF/data/TrainTest_tipburn/fm_crops_combined",
    fvfm_dir="/lustre/BIF/nobackup/to001/thesis_MBF/data/TrainTest_tipburn/fvfm_crops_combined",
    train_frac=0.75,
    transform=transforms,
    seed=42,
    give_name=False,
)

img, mask = dataset[0]

import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 4, sharex=True, sharey=True)
axes[0].imshow(np.moveaxis(img[:3], 0, 2).astype(np.uint8))
axes[1].imshow(img[3])
axes[2].imshow(img[4])
axes[3].imshow(mask)
plt.show()

# Test after applying masks to fluorescence images


# Define data loaders for training and testing
def get_loaders(
    dataset,
    img_dir,
    label_dir,
    train_augs,
    test_augs,
    batch_size,
    num_workers,
    train_frac=0.75,
    pin_memory=True,
    seed=42,
):
    """Creates PyTorch DataLoaders for train and test dataset.

    Args:
        dataset (torch.utils.data.Dataset): Dataset class inherited from PyTorch's Dataset class.
        img_dir (string): Path of directory containing the image data.
        label_dir (string): Path of directory containing the mask_paths of the image data.
        train_augs (albumentations.Compose/transforms.Compose): Albumentations or PyTorch transforms for train.
        test_augs (albumentations.Compose/transforms.Compose): Albumentations or PyTorch transforms for test.
        batch_size (int): Number of samples in each batch.
        num_workers (int): Number of worker processes for data loading.
        train_frac (float, optional): Fraction of data to be used for training. Defaults to 0.75.
        pin_memory (bool, optional): Speeds up data transfer from CPU to GPU. Defaults to True.
        seed (int, optional): Seed for reproducible train test split of data. Defaults to 42.

    Returns:
        _type_: _description_
    """
    # Get train and test datasets
    train_ds = dataset(
        img_dir=img_dir,
        label_dir=label_dir,
        train_frac=train_frac,
        is_train=True,
        transform=train_augs,
        seed=seed,
    )
    test_ds = dataset(
        img_dir=img_dir,
        label_dir=label_dir,
        train_frac=train_frac,
        is_train=False,
        transform=test_augs,
        seed=seed,
    )

    # Create DataLoaders of datasets
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, test_loader


# MNIST handwritten digit dataset for testing classification
def MNIST_digit_loaders(batch_size, num_workers, pin_memory=True):
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            "/lustre/BIF/nobackup/to001/thesis_MBF/data",
            train=True,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,)),
                ]
            ),
        ),
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            "/lustre/BIF/nobackup/to001/thesis_MBF/data",
            train=False,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,)),
                ]
            ),
        ),
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, test_loader
