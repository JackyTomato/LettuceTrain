import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from PIL import Image
import os


class LeafDataset(torch.utils.data.Dataset):
    def __init__(self, directory, is_train=True, transforms=None):
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
        return len(self.images)

    def __getitem__(self, idx):
        # Retrieve image and label
        img = d2l.Image.open(self.images[idx])
        labels = torch.tensor(int(self.labels[idx]), dtype=torch.float32)

        # Assertions to check that everything is correct
        assert isinstance(img, Image), "Image variable should be a PIL Image"
        assert isinstance(
            labels, torch.Tensor
        ), "Labels varibable should be a torch tensor"
        assert (
            labels.dtype == torch.float32
        ), "Labels variable datatype should be float32"

        if self.transforms is not None:
            img = self.transforms(img)

        return img, labels


# Define data loaders for training and testing
def load_train_data(augs, batch_size):
    dataset = LeafDataset(root, is_train=True, transforms=augs)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    return dataloader


def load_test_data(augs, batch_size):
    dataset = LeafDataset(root, is_train=False, transforms=augs)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )
    return dataloader
