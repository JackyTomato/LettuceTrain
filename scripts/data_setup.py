#!/usr/bin/env python3
"""
Functionality for creating lettuce dataset as PyTorch Dataset
and loading thre train & test datasets.

TODO:
    - Implement pre TB dataset
"""
# Import statements
import os
import torch
import numpy as np
import re
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torchvision import transforms, datasets
from PIL import Image
from sklearn.model_selection import train_test_split, KFold
from skimage.transform import resize

# Import supporting modules
import utils


# Define Pytorch Dataset class for lettuce segmentation dataset
class LettuceSegDataset(Dataset):
    def __init__(
        self,
        img_dir,
        label_dir,
        is_train,
        fm_dir=None,
        fvfm_dir=None,
        train_frac=0.75,
        kfold=None,
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

        Allows for a training fraction of 1, in that case the whole dataset is provided.
        When doing K-fold cross valiation, the whole dataset is also provided,
        as the split will be performed downstream.

        Args:
            img_dir (str): Filepath of directory containing the images.
            label_dir (str): Filepath of directory  containing the segmentation masks.
            is_train (bool): If true, gives train data. If false, gives test data.
            fm_dir (str, optional): Filepath of directory containing Fm images for fusion. Defaults to None.
            fvfm_dir (str, optional): Filepath of directory containing FvFm images for fusion. Defaults to None.
            train_frac (float, optional): Fraction of data that is train. Defaults to 0.75.
            kfold (int, optional): K for K-fold cross validation. If not None, train_frac is ignored. Defaults to None.
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
        else:
            incomplete_masks = False

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

        # Split into train and test sets if desired when not doing K-fold cross validation
        if (train_frac < 1) and (kfold is None):
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

        # Don't split when training fraction is 1 or when doing K-fold cross validation
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
        # Check if Fm and FvFm should be included
        fm_exists = hasattr(self, "fm_paths")
        fvfm_exists = hasattr(self, "fvfm_paths")

        # Retrieve image, should be np.array for albumentations.transforms
        img = np.array(Image.open(self.img_paths[index]))

        # Also retrieve Fm and FvFm images if desired
        if fm_exists:
            fm = np.array(Image.open(self.fm_paths[index]))
            fm = fm / fm.max() * 255  # normalize Fm values as they are large
            img_fit = resize(img, (fm.shape[0], fm.shape[1], 3), anti_aliasing=True)
            fm = fm * (img_fit.sum(axis=2) > 0)  # apply RGB background mask
        if fvfm_exists:
            fvfm = np.array(Image.open(self.fvfm_paths[index]))
            img_fit = resize(img, (fvfm.shape[0], fvfm.shape[1], 3), anti_aliasing=True)
            fvfm = fvfm * (img_fit.sum(axis=2) > 0)  # apply RGB background mask

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
            if fm_exists or fvfm_exists:
                # Apply augmentations
                grayscales = [mask]
                if fm_exists:
                    grayscales.append(fm)
                if fvfm_exists:
                    grayscales.append(fvfm)
                augmentations = self.transform(image=img, masks=grayscales)

                # Retrieve augmentations
                if (fm_exists) and (not fvfm_exists):
                    mask, fm = augmentations["masks"]
                elif (not fm_exists) and (fvfm_exists):
                    mask, fvfm = augmentations["masks"]
                elif (fm_exists) and (fvfm_exists):
                    mask, fm, fvfm = augmentations["masks"]
            else:
                # Apply augmentations
                augmentations = self.transform(image=img, mask=mask)

                # Retrieve augmentations
                mask = augmentations["mask"]
            img = augmentations["image"]

        # Compile resulting images, in a way suitable for fusion if desired
        if fm_exists:
            img = np.concatenate([img, fm[np.newaxis, :, :]], axis=0)
        if fvfm_exists:
            img = np.concatenate([img, fvfm[np.newaxis, :, :]], axis=0)
        result = (img, mask)

        # Also provide image name if desired
        if self.give_name:
            img_name = self.img_names[index].split(os.extsep)[0]
            result.append(img_name)

        return result


# Create PyTorch dataset class for lettuce pretipburn classification
class LettucePreTBClassDataset(Dataset):
    def __init__(
        self,
        img_dir,
        is_train,
        fm_dir=None,
        fvfm_dir=None,
        train_frac=0.75,
        kfold=None,
        transform=None,
        seed=42,
        give_name=False,
    ):
        """Creates a PyTorch Dataset class of the lettuce pre-tipburn classification dataset.

        Uses sorted() to match filenames of images and masks. Check if filenames
        are constructed in such a way that sorted() will sort the filenames in the same way.
        Otherwise images and masks will be mismatched when loading the data.

        Fm and FvFm images can be included as additional input. The Fm and FvFm are stacked
        with the RGB images as additional channels.

        Allows for a training fraction of 1, in that case the whole dataset is provided.
        When doing K-fold cross valiation, the whole dataset is also provided,
        as the split will be performed downstream.

        Args:
            img_dir (str): Filepath of directory containing the images.
            is_train (bool): If true, gives train data. If false, gives test data.
            fm_dir (str, optional): Filepath of directory containing Fm images for fusion. Defaults to None.
            fvfm_dir (str, optional): Filepath of directory containing FvFm images for fusion. Defaults to None.
            train_frac (float, optional): Fraction of data that is train. Defaults to 0.75.
            kfold (int, optional): K for K-fold cross validation. If not None, train_frac is ignored. Defaults to None.
            transform (albumentations.Compose, optional): Transformations for data aug. Defaults to None.
            seed (int, optional): Seed for reproducible train test split of data. Defaults to 42.
            give_name (bool, optional): If True, dataset also provides image name.
        """
        self.transform = transform
        self.give_name = give_name

        # List all image and optionally Fm and FvFm filenames
        self.img_names = sorted(os.listdir(img_dir))
        if fm_dir is not None:
            fm_names = sorted(os.listdir(fm_dir))
        if fvfm_dir is not None:
            fvfm_names = sorted(os.listdir(fvfm_dir))

        # Create lists of filepaths for images, corresponding list of GT labels and find indices to filter out
        img_paths = []
        labels = []
        regex_exp = re.compile("^\d{2}")
        regex_trayID = re.compile(".+Tray_(\d{2,})")
        ind_filter = []
        regex_time = re.compile("^\d{2}-(\d+)-")
        for ind, img_name in enumerate(self.img_names):
            # List image name paths
            img_path = os.path.join(img_dir, img_name)
            img_paths.append(img_path)
            # List corresponding labels based on timepoint and trayID in filename
            match_exp = regex_exp.match(img_name)
            exp = match_exp.group(1)
            match_trayID = regex_trayID.match(img_name)
            trayID = int(match_trayID.group(1))
            if exp == 41:
                if trayID > 80:
                    label = 1
                else:
                    label = 0
            elif exp == 51:
                if trayID <= 80:
                    label = 1
                else:
                    label = 0
            labels.append(label)
            # List indices to filter out if they are experiment 41 or later timepoints of experiment 51
            if ind.startswith("41"):
                ind_filter.append(ind)
            else:
                match_time = regex_time.match(img_name)
                time = int(match_time.group(1))
                if time > 24:
                    ind_filter.append(ind)
        # Filter out RGB images and labels from experiment 41 or later timepoints of experiment 51
        img_paths = [
            path for ind, path in enumerate(img_paths) if ind not in ind_filter
        ]
        labels = [label for ind, label in enumerate(labels) if ind not in ind_filter]

        # Also create lists of filepaths for Fm and FvFm if desired
        if fm_dir is not None:
            fm_paths = []
            for fm_name in fm_names:
                fm_path = os.path.join(fm_dir, fm_name)
                fm_paths.append(fm_path)
            # Filter out RGB images from experiment 41 or later timepoints of experiment 51
            fm_paths = [
                path for ind, path in enumerate(fm_paths) if ind not in ind_filter
            ]
        if fvfm_dir is not None:
            fvfm_paths = []
            for fvfm_name in fvfm_names:
                fvfm_path = os.path.join(fvfm_dir, fvfm_name)
                fvfm_paths.append(fvfm_path)
            # Filter out RGB images from experiment 41 or later timepoints of experiment 51
            fvfm_paths = [
                path for ind, path in enumerate(fvfm_paths) if ind not in ind_filter
            ]

        # Split into train and test sets if desired when not doing K-fold cross validation
        if (train_frac < 1) and (kfold is None):
            if (fm_dir is None) and (fvfm_dir is None):
                split = train_test_split(
                    img_paths,
                    labels,
                    train_size=train_frac,
                    random_state=seed,
                )

                # Give train or test data as requested
                if is_train:
                    self.img_paths = split[0]
                    self.labels = split[2]
                else:
                    self.img_paths = split[1]
                    self.labels = split[3]
            elif (fm_dir is not None) and (fvfm_dir is None):
                split = train_test_split(
                    img_paths,
                    fm_paths,
                    labels,
                    train_size=train_frac,
                    random_state=seed,
                )

                # Give train or test data as requested
                if is_train:
                    self.img_paths = split[0]
                    self.fm_paths = split[2]
                    self.labels = split[4]
                else:
                    self.img_paths = split[1]
                    self.fm_paths = split[3]
                    self.labels = split[5]
            elif (fm_dir is None) and (fvfm_dir is not None):
                split = train_test_split(
                    img_paths,
                    fvfm_paths,
                    labels,
                    train_size=train_frac,
                    random_state=seed,
                )

                # Give train or test data as requested
                if is_train:
                    self.img_paths = split[0]
                    self.fvfm_paths = split[2]
                    self.labels = split[4]
                else:
                    self.img_paths = split[1]
                    self.fvfm_paths = split[3]
                    self.labels = split[5]
            elif (fm_dir is not None) and (fvfm_dir is not None):
                split = train_test_split(
                    img_paths,
                    fm_paths,
                    fvfm_paths,
                    labels,
                    train_size=train_frac,
                    random_state=seed,
                )

                # Give train or test data as requested
                if is_train:
                    self.img_paths = split[0]
                    self.fm_paths = split[2]
                    self.fvfm_paths = split[4]
                    self.labels = split[6]
                else:
                    self.img_paths = split[1]
                    self.fm_paths = split[3]
                    self.fvfm_paths = split[5]
                    self.labels = split[7]

        # Don't split when training fraction is 1 or when doing K-fold cross validation
        else:
            self.img_paths = img_paths
            self.labels = labels
            if fm_dir is not None:
                self.fm_paths = fm_paths
            if fvfm_dir is not None:
                self.fvfm_paths = fvfm_paths

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        # Check if Fm and FvFm should be included
        fm_exists = hasattr(self, "fm_paths")
        fvfm_exists = hasattr(self, "fvfm_paths")

        # Retrieve image, should be np.array for albumentations.transforms
        img = np.array(Image.open(self.img_paths[index]))

        # Also retrieve Fm and FvFm images if desired
        if fm_exists:
            fm = np.array(Image.open(self.fm_paths[index]))
            fm = fm / fm.max() * 255  # normalize Fm values as they are large
            img_fit = resize(img, (fm.shape[0], fm.shape[1], 3), anti_aliasing=True)
            fm = fm * (img_fit.sum(axis=2) > 0)  # apply RGB background mask
        if fvfm_exists:
            fvfm = np.array(Image.open(self.fvfm_paths[index]))
            img_fit = resize(img, (fvfm.shape[0], fvfm.shape[1], 3), anti_aliasing=True)
            fvfm = fvfm * (img_fit.sum(axis=2) > 0)  # apply RGB background mask

        # Retrieve mask, mask could be .json or an image format
        size = img.shape[:2]
        if self.labels[index].endswith(".json"):
            mask = utils.binary_poly2px(self.labels[index], custom_size=size)
        elif self.labels[index].endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff")):
            mask = np.array(Image.open(self.labels[index]))
        else:
            mask = np.zeros(size, dtype=np.int32)  # Create empty mask for missing mask
        if 255.0 in mask:
            mask[mask == 255.0] = 1.0

        # Apply data augmentation transforms to image, mask and optionally Fm and FvFm
        if self.transform is not None:
            if fm_exists or fvfm_exists:
                # Apply augmentations
                grayscales = [mask]
                if fm_exists:
                    grayscales.append(fm)
                if fvfm_exists:
                    grayscales.append(fvfm)
                augmentations = self.transform(image=img, masks=grayscales)

                # Retrieve augmentations
                if (fm_exists) and (not fvfm_exists):
                    mask, fm = augmentations["masks"]
                elif (not fm_exists) and (fvfm_exists):
                    mask, fvfm = augmentations["masks"]
                elif (fm_exists) and (fvfm_exists):
                    mask, fm, fvfm = augmentations["masks"]
            else:
                # Apply augmentations
                augmentations = self.transform(image=img, mask=mask)

                # Retrieve augmentations
                mask = augmentations["mask"]
            img = augmentations["image"]

        # Compile resulting images, in a way suitable for fusion if desired
        if fm_exists:
            img = np.concatenate([img, fm[np.newaxis, :, :]], axis=0)
        if fvfm_exists:
            img = np.concatenate([img, fvfm[np.newaxis, :, :]], axis=0)
        result = (img, mask)

        # Also provide image name if desired
        if self.give_name:
            img_name = self.img_names[index].split(os.extsep)[0]
            result.append(img_name)

        return result


# Define data loaders for training and testing
def get_loaders(
    dataset,
    img_dir,
    label_dir,
    train_augs,
    test_augs,
    batch_size,
    num_workers,
    fm_dir=None,
    fvfm_dir=None,
    train_frac=0.75,
    kfold=None,
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
        fm_dir (str, optional): Filepath of directory containing Fm images for channel stacked input.
        fvfm_dir (str, optional): Filepath of directory containing FvFm images for channel stacked input.
        train_frac (float, optional): Fraction of data to be used for training. Defaults to 0.75.
        kfold (int, optional): K for K-fold cross validation. If not None, train_frac is ignored. Defaults to None.
        pin_memory (bool, optional): Speeds up data transfer from CPU to GPU. Defaults to True.
        seed (int, optional): Seed for reproducible train test split of data. Defaults to 42.

    Returns:
        tuple: Contains torch.utils.data.DataLoader objects for training and testing dataset.
        Or when doing K-fold cross validation:
        list: Containing tuples of torch.utils.data.DataLoader objects for training and testing dataset.
    """
    if kfold is None:
        # Get train and test datasets
        train_ds = dataset(
            img_dir=img_dir,
            label_dir=label_dir,
            train_frac=train_frac,
            fm_dir=fm_dir,
            fvfm_dir=fvfm_dir,
            is_train=True,
            transform=train_augs,
            seed=seed,
        )
        test_ds = dataset(
            img_dir=img_dir,
            label_dir=label_dir,
            train_frac=train_frac,
            fm_dir=fm_dir,
            fvfm_dir=fvfm_dir,
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
        loaders = (train_loader, test_loader)

    # Create dataloaders for each split of K-fold cross validation
    else:
        # Get full to be split into K-folds dataset
        full_ds = dataset(
            img_dir=img_dir,
            label_dir=label_dir,
            train_frac=train_frac,
            kfold=kfold,
            fm_dir=fm_dir,
            fvfm_dir=fvfm_dir,
            is_train=True,
            transform=train_augs,
            seed=seed,
        )

        # Create generator of train and set indices for each fold
        kfolder = KFold(n_splits=kfold, shuffle=True, random_state=seed)
        inds_kfold = kfolder.split(full_ds)

        # Create list of train and test DataLoader objects
        loaders = []
        for train_inds, test_inds in inds_kfold:
            # Create DataLoaders for current fold and add to list
            train_loader = DataLoader(
                full_ds,
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=pin_memory,
                sampler=train_inds,
            )
            test_loader = DataLoader(
                full_ds,
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=pin_memory,
                sampler=test_inds,
            )
            loaders.append((train_loader, test_loader))

    return loaders


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
