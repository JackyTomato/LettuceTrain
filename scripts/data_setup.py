import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split


class LettuceDataset(Dataset):
    def __init__(self, directory, is_train=True, transforms=None):
        """Creates lettuce dataset as a PyTorch DataSet class object

        Args:
            directory (str): Path to file containing image names and labels.
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
            self.images, self.labels, test_size=0.25, random_state=42
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
def load_train_data(augs, batch_size):
    train_dataset = LettuceDataset(root, is_train=True, transforms=augs)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    return train_dataloader


def load_test_data(augs, batch_size):
    test_dataset = LettuceDataset(root, is_train=False, transforms=augs)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )
    return test_dataloader
