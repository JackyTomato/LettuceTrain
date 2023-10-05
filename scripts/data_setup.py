"""
Functionality for creating lettuce dataset as PyTorch Dataset
and loading thre train & test datasets.

TODO:
    - Make dataset class for lettuce
    - Allow for separate transforms for train and test
    - Test to see if it properly loads dataets
"""
# Import statements
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from PIL import Image
from sklearn.model_selection import train_test_split


# Define Pytorch Dataset class for lettuce dataset
class LettuceDataset(Dataset):
    def __init__(
        self, directory, train_frac=0.75, is_train=True, transforms=None, seed=42
    ):
        """Creates lettuce dataset as a PyTorch Dataset class object

        Args:
            directory (str): Path to file containing image names and labels.
            train_frac (float): Proportion of dataset to be training data.
            is_train (bool, optional): _description_. Defaults to True.
            transforms (transforms.Compose, optional): Composed torchvision transformations. Defaults to None.
        """
        self.images = []
        self.labels = []
        self.transforms = transforms

        with open(os.path.join(root, "Leaf_counts.csv"), "r") as f:
            for line in f:
                filename, n_leafs = line.rstrip().split(", ")
                filename = filename + "_rgb.png"
                img_path = os.path.join(directory, filename)
                # Fill the corresponding lists with the images and labels
                self.images.append(img_path)
                self.labels.append(n_leafs)

        X_train, X_test, y_train, y_test = train_test_split(
            self.images, self.labels, train_size=train_frac, random_state=seed
        )
        if is_train:
            self.images = X_train
            self.labels = y_train
        else:
            self.images = X_test
            self.labels = y_test

    def __len__(self):
        """Returns number of images in dataset

        Returns:
            int: Length of vector of images
        """
        return len(self.images)

    def __getitem__(self, idx):
        """Returns images and labels in dataset

        Args:
            idx (_type_): _description_

        Returns:
            _type_: _description_
        """
        # Retrieve image and label
        img = Image.open(self.images[idx])
        labels = torch.tensor(int(self.labels[idx]), dtype=torch.float32)

        # Assertions to check that everything is correct
        assert isinstance(img, Image), "Image variable should be a PIL Image"
        assert isinstance(
            labels, torch.Tensor
        ), "Labels variable should be a torch tensor"
        assert (
            labels.dtype == torch.float32
        ), "Labels variable datatype should be float32"

        if self.transforms is not None:
            img = self.transforms(img)

        return img, labels


# Define data loaders for training and testing
def get_loaders(
    dataset,
    img_dir,
    label_dir,
    augs,
    batch_size,
    num_workers,
    train_frac=0.75,
    pin_memory=True,
):
    """Creates PyTorch DataLoaders for train and test dataset.

    Args:
        dataset (torch.utils.data.Dataset): Dataset class inherited from PyTorch's Dataset class.
        img_dir (string): Path of directory containing the image data.
        label_dir (string): Path of directory containing the labels of the image data.
        augs (albumentations.Compose/transforms.Compose): Albumentations or PyTorch transforms (composed).
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
        transform=augs,
        train_frac=train_frac,
        is_train=True,
    )
    test_ds = dataset(
        img_dir=img_dir,
        label_dir=label_dir,
        transform=augs,
        train_frac=train_frac,
        is_train=False,
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
