#!/usr/bin/env python3
"""
Contains various utility functions such as checkpointing and performance metrics.

TODO:
    - Add segmentation performance functions (if we cant just use PyTorch's)
    - Use os.path.splitext to clean up save filenames
    - Add functions for plotting (plot data augs, plot predictions)
    - Convert polygons to pixel masks
"""
# Import functions
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import albumentations as A
import numpy as np
import json
import cv2
from torchinfo import summary
from pathlib import Path
from shutil import copyfile
from skimage import io
from PIL import Image


# Checkpointing
def save_checkpoint(state, target_dir, model_name):
    """Saves a PyTorch model state(s) to a target directory.

    Args:
        state (nn.Module): State(s) of a PyTorch model to save.
        target_dir (str): A directory for saving the model to.
        model_name (str): A filename for the saved model. Should include
        ".pth", ".pt", ".pth.tar" or "pt.tar" as the file extension.

    Example usage:
        save_model(state=model,
                    target_dir="models",
                    model_name="tipburn_resnet50_classifier.pth.tar")
    """
    # Create target directory
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)

    # Create model save path
    assert model_name.endswith(
        (".pth", ".pt", ".pth.tar", "pt.tar")
    ), "model_name should end with '.pth', '.pt', '.pth.tar' or 'pt.tar'"
    model_save_path = os.path.join(target_dir_path, model_name)

    # Save the model state_dict()
    torch.save(obj=state, f=model_save_path)
    print(f"[INFO] Saved model states to {model_save_path}")


def load_checkpoint(checkpoint, model):
    """Loads a saved PyTorch model state.

    Args:
        checkpoint (str): File of saved model state including path if necessary.
        model (nn.Module): PyTorch model on which the state will be applied.
    """
    model.load_state_dict(torch.load(checkpoint)["state_dict"])
    print(f"[INFO] Loaded model state {checkpoint}")


# Loss functions
class DiceBCEWithLogitsLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        """Combination of Dice and BCE loss function.

        Adapted from:
            https://www.kaggle.com/code/bigironsphere/loss-function-library-keras-pytorch#BCE-Dice-Loss

        Dice and BCE are combined to include both Dice's accuracy and BCE's stability.
        Additionally, according to Ma et al. (2021), compound loss functions tend to be more robust:
            https://doi.org/10.1016/j.media.2021.102035
            https://github.com/JunMa11/SegLoss

        Args:
            weight (optional): Irrelevant for loss function. Defaults to None.
            size_average (optional): Irrelevant for loss function. Defaults to True.
        """
        super(DiceBCEWithLogitsLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # Convert logits to probabilities
        prob_inputs = F.sigmoid(inputs)

        # Flatten label and prediction tensors
        flat_inputs = prob_inputs.view(-1)
        flat_targets = targets.view(-1)

        # Calculate Dice
        intersection = (flat_inputs * flat_targets).sum()
        dice_loss = 1 - (2.0 * intersection + smooth) / (
            flat_inputs.sum() + flat_targets.sum() + smooth
        )

        # Calculate BCE
        BCE_fn = nn.BCEWithLogitsLoss()
        BCE = BCE_fn(inputs, targets)

        # Sum Dice and BCE
        Dice_BCE = BCE + dice_loss
        return Dice_BCE


# Performance metrics
def class_accuracy(pred_logits, labels):
    """Calculate classification accuracy.

    Accuracy is determined as the fraction of predictions that match
    the ground-truth labels.

    Args:
        pred_logits (torch.Tensor): Tensor of prediction logits.
        labels (torch.Tensor): Tensor of ground truth labels.

    Returns:
        A float of classification accuracy. For example:
            0.8317
    """
    # Check if predictions and labels are of the same length
    assert len(pred_logits) == len(
        labels
    ), "Arguments pred_logits and labels should be tensors of the same length"

    # Get predicted labels
    pred_labels = pred_logits.argmax(dim=1)

    # Calculate and return accuracy
    pred_acc = (pred_labels == labels).sum().item() / len(pred_labels)
    return pred_acc


# .json parser for config.json files
def parse_json(filepath):
    """Parses the a .json file as a dictionary containing all the values.

    Args:
        filepath (str): File path to the .json file.
    """
    # Parse .json files as dict with json's load function
    with open(filepath, "r") as json_file:
        json_dict = json.load(json_file)
    return json_dict


# Save training results, network summaries and config
def save_train_results(dict_results, target_dir, filename):
    """Writes training loop results to a tab-delimited text file.

    Input dictionary of results should be in format as constructed in train.py.
    Thus, with "epoch", "train_loss", "train_perform", "test_loss" and "test_perform"
    as keys.

    Args:
        dict_results (dict): Dictionary of training loop results.
        target_dir (str): Target directory in which to write the file.
        filename (str): File name of the text file to be written.
    """
    # Create target directory
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)

    # Sort keys to get sorted columns
    sorted_keys = [
        "epoch",
        "train_loss",
        "train_perform",
        "test_loss",
        "test_perform",
    ]

    # Get number of items to write rows
    num_items = len(dict_results["epoch"])

    # Write dictonary to file
    filepath = os.path.join(target_dir, filename)
    with open(filepath, "w") as f:
        # Write header
        header = "\t".join(sorted_keys)
        f.write(header)
        f.write("\n")

        # Write rows
        for row in range(num_items):
            line_values = []
            for key in sorted_keys:
                target_column = dict_results[key]
                line_values.append(str(target_column[row]))
            line = "\t".join(line_values)
            f.write(line)
            f.write("\n")
    print(f"[INFO] Saved training results to {filepath}")


def save_network_summary(model, target_dir, filename, n_channels=3):
    """Writes a torchinfo summary and raw print summary of a model to a text file

    Args:
        model (nn.Module): PyTorch model to be summarized.
        target_dir (str): Target directory in which to write the file.
        filename (str): File name of the text file to be written.
        n_channels (int, optional): Number of input channels to test. Defaults to 3.
    """
    # Create target directory
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)

    # Get torchinfo summary of model
    model_stats = summary(
        model=model,
        input_size=(1, n_channels, 512, 512),
        col_names=["kernel_size", "input_size", "output_size", "num_params"],
        verbose=0,
    )
    str_model_stats = str(model_stats)

    # Get raw model as str (show all layers and parameters)
    str_raw_model = str(model)

    # Write model summaries to file
    filepath = os.path.join(target_dir, filename)
    with open(filepath, "w") as f:
        f.write("[TORCHINFO SUMMARY]")
        f.write("\n")
        f.write(str_model_stats)
        f.write("\n\n")
        f.write("[RAW SUMMARY]")
        f.write("\n")
        f.write(str_raw_model)
    print(f"[INFO] Saved network summaries to {filepath}")


def save_config(target_dir, filename, config_name="config.json"):
    """Saves config.json used for the model to a new text file.

    Args:
        target_dir (str): Target directory in which to write the file.
        filename (str): File name of the text file to be written.
        config_name (str): Name of config file to save. Defaults to "config.json".
    """
    # Create target directory
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)

    # Copy config to new file
    filepath = os.path.join(target_dir, filename)
    copyfile(config_name, filepath)
    print(f"[INFO] Saved config.json to {filepath}")


def save_img(img, target_dir, filename):
    """Saves np.ndarray or PIL.Image.Image image object to a file.

    Args:
        img (np.ndarray/PIL.Image.Image): Image object to be saved.
        target_dir (str): Target directory in which to save file.
        filename (str): Name of file to which the image is saved.

    Raises:
        Exception: Image should be either a np.ndarray or a PIL.Image.Image.
    """
    # Create target directory
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)

    # Create filepath for writing the image file
    filepath = os.path.join(target_dir, filename)

    # Save img with skimage if np.ndarray
    if type(img) is np.ndarray:
        io.imsave(filepath, img, check_contrast=False)

    # Save img with PIL if Image
    elif type(img) is Image.Image:
        img.save(filepath)

    else:
        raise Exception("Input image object should be np.ndarray or PIL.Image.Image")


# Read Fluorcam's .fimg files
def read_fimg(filepath):
    """Reads Fluorcam .fimg files as a np.ndarray with float32 values.

    Float32 is used to normalize bit values to values from -1 to 1.

    Args:
        filepath (str): Filepath of the .fimg file to be read.

    Returns:
        np.ndarray: Image as np.ndarray with height, width of 1024, 1360.
    """
    with open(filepath) as fimg:
        # Convert to np.array with float32 for normalized values
        img = np.fromfile(fimg, np.dtype("float32"))

        # Delete first two values (info about image dimensions)
        img = img[2:]

        # Reshape to intended dimensions
        img = np.reshape(img, newshape=(1024, 1360))

        # Remove negative artifacts in Fv/Fm
        img[img < 0] = 0
        return img


# Convert binary polygon .json files to pixel masks
def binary_poly2px(filepath):
    """Converts binary polygon mask in .json format from filepath to pixel mask.

    Args:
        filepath (str): Filepath of .json polygon mask.

    Returns:
        np.ndarray: Binary pixel mask of ints where 0 is background and 1 is the annotation.
    """
    # Open .json and extract points
    poly_json = json.load(open(filepath, "r"))
    poly_points = poly_json["shapes"][0]["points"]
    poly_points = np.array(poly_points, dtype=np.int32)

    # Create empty mask to draw on
    mask = np.zeros((poly_json["imageHeight"], poly_json["imageWidth"]))

    # Draw polygon on empty mask
    cv2.fillPoly(mask, [poly_points], color=1)
    return mask.astype(np.int32)
