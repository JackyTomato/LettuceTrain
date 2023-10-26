#!/usr/bin/env python3
"""
Functionality for creating lettuce dataset as PyTorch Dataset
and loading thre train & test datasets.

TODO:
    - Test to see if it properly loads lettuce data
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
        self, img_dir, label_dir, is_train, train_frac=0.75, transform=None, seed=42
    ):
        """Creates a PyTorch Dataset class of the lettuce segmantation dataset.

        Uses sorted() to match filenames of images and masks. Check if filenames
        are constructed in such a way that sorted() will sort the filenames in the same way.
        Otherwise images and masks will be mismatched when loading the data.

        Args:
            img_dir (str): Filepath of directory containing the images.
            label_dir (str): Filepath of directory  containing the segmentation masks.
            is_train (bool): If true, gives train data. If false, gives test data.
            train_frac (float, optional): Fraction of data that is train. Defaults to 0.75.
            transform (albumentations.Compose, optional): Transformations for data aug. Defaults to None.
            seed (int, optional): Seed for reproducible train test split of data. Defaults to 42.
        """
        self.transform = transform

        # List all image and mask filenames
        img_names = sorted(os.listdir(img_dir))
        mask_names = sorted(os.listdir(label_dir))

        # Create lists of filepath for images and masks
        img_paths = []
        mask_paths = []
        for img_name, mask_name in zip(img_names, mask_names):
            img_path = os.path.join(img_dir, img_name)
            mask_path = os.path.join(label_dir, mask_name)
            img_paths.append(img_path)
            mask_paths.append(mask_path)

        # Split train and test sets
        img_train, img_test, mask_train, mask_test = train_test_split(
            img_paths, mask_paths, train_size=train_frac, random_state=seed
        )

        # Give train or test data as requested
        if is_train:
            self.img_paths = img_train
            self.mask_paths = mask_train
        else:
            self.img_paths = img_test
            self.mask_paths = mask_test

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        # Retrieve image and mask, should be np.array for albumentations.transforms
        img = np.array(Image.open(self.img_paths[index]), dtype=np.float32)
        if self.mask_paths[index].endswith(".json"):
            mask = utils.binary_poly2px(self.mask_paths[index]).astype(np.float32)
        else:
            mask = np.array(Image.open(self.mask_paths[index]), dtype=np.float32)
        if 255.0 in mask:
            mask[mask == 255.0] = 1.0

        # Apply data augmentation transforms to image and mask
        if self.transform is not None:
            augmentations = self.transform(image=img, mask=mask)
            img = augmentations["image"]
            mask = augmentations["mask"]

        return img, mask


class LettuceDataset(Dataset):
    def __init__(
        self, directory, train_frac=0.75, is_train=True, transforms=None, seed=42
    ):
        """Creates lettuce dataset as a PyTorch Dataset class object

        Args:
            directory (str): Path to file containing image names and mask_paths.
            train_frac (float): Proportion of dataset to be training data.
            is_train (bool, optional): _description_. Defaults to True.
            transforms (transforms.Compose, optional): Composed torchvision transformations. Defaults to None.
        """
        self.images = []
        self.mask_paths = []
        self.transforms = transforms

        with open(os.path.join(root, "Leaf_counts.csv"), "r") as f:
            for line in f:
                filename, n_leafs = line.rstrip().split(", ")
                filename = filename + "_rgb.png"
                img_path = os.path.join(directory, filename)
                # Fill the corresponding lists with the images and mask_paths
                self.images.append(img_path)
                self.mask_paths.append(n_leafs)

        X_train, X_test, y_train, y_test = train_test_split(
            self.images, self.mask_paths, train_size=train_frac, random_state=seed
        )
        if is_train:
            self.images = X_train
            self.mask_paths = y_train
        else:
            self.images = X_test
            self.mask_paths = y_test

    def __len__(self):
        """Returns number of images in dataset

        Returns:
            int: Length of vector of images
        """
        return len(self.images)

    def __getitem__(self, idx):
        """Returns images and mask_paths in dataset

        Args:
            idx (_type_): _description_

        Returns:
            _type_: _description_
        """
        # Retrieve image and label
        img = Image.open(self.images[idx])
        mask_paths = torch.tensor(int(self.mask_paths[idx]), dtype=torch.float32)

        # Assertions to check that everything is correct
        assert isinstance(img, Image), "Image variable should be a PIL Image"
        assert isinstance(
            mask_paths, torch.Tensor
        ), "Labels variable should be a torch tensor"
        assert (
            mask_paths.dtype == torch.float32
        ), "Labels variable datatype should be float32"

        if self.transforms is not None:
            img = self.transforms(img)

        return img, mask_paths


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
    )
    test_ds = dataset(
        img_dir=img_dir,
        label_dir=label_dir,
        train_frac=train_frac,
        is_train=False,
        transform=test_augs,
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
