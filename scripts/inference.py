#!/usr/bin/env python3
"""
Runs inference of models. Script can save results and corresponding performance.

TODO:
    - Remove convoluted method of sending model & data to device as normal method works now
"""

# Import statements
import os
import torch
import torch.nn as nn
import torchvision.transforms.functional as F
import torchmetrics
import albumentations as A
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage import util
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import make_grid, draw_segmentation_masks, save_image
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from pathlib import Path

# Import supporting modules
import data_setup, model_builder, utils


# Load model from saved model filename
def load_model(model_filepath, device="cuda", multi_gpu=False):
    """Loads a model from the filepath of the saved model.

    The model is loaded by parsing the corresponding saved config .json,
    using the config settings to initialize the model and subsequently
    loading the saved model state into the initialized model.

    Assumes that the saved config .json is in the same directory
    as the saved model. Additionally, assumes that the saved config
    is named "config_<model filename without file extension>.json".

    Args:
        model_filepath (str): Filepath of saved model states.
        device (str, optional): Device to send model to, "cpu" or "cuda". Defaults to "cuda".
        multi_gpu (bool): If True, loads model on multiple GPUs, assuming device is "cuda".

    Returns:
        torch.nn.Module: Loaded model as a PyTorch nn.Module class.
    """
    model_dir, model_filename = os.path.split(model_filepath)

    # Parse config.json as dict
    config_filename = f"config_{model_filename.split(os.extsep)[0]}.json"
    config_path = os.path.join(model_dir, config_filename)
    config_dict = utils.parse_json(config_path)

    # Assign model setings from config
    MODEL_TYPE = eval(config_dict["MODEL_TYPE"])
    MODEL_NAME = config_dict["MODEL_NAME"]
    ENCODER_NAME = config_dict["ENCODER_NAME"]
    ENCODER_WEIGHTS = config_dict["ENCODER_WEIGHTS"]
    if ENCODER_WEIGHTS == "None":  # Allow for untrained encoder
        ENCODER_WEIGHTS = eval(ENCODER_WEIGHTS)
    N_CHANNELS = config_dict["N_CHANNELS"]
    N_CLASSES = config_dict["N_CLASSES"]
    DECODER_ATTENTION = config_dict["DECODER_ATTENTION"]
    if DECODER_ATTENTION == "None":  # Allow for no decoder attention
        DECODER_ATTENTION = eval(DECODER_ATTENTION)
    ENCODER_FREEZE = eval(config_dict["ENCODER_FREEZE"])
    print(f"[INFO] Loading config.json was succesful!")

    # Initialize model and send to device
    model = MODEL_TYPE(
        model_name=MODEL_NAME,
        encoder_name=ENCODER_NAME,
        encoder_weights=ENCODER_WEIGHTS,
        n_channels=N_CHANNELS,
        n_classes=N_CLASSES,
        decoder_attention=DECODER_ATTENTION,
        encoder_freeze=ENCODER_FREEZE,
    )
    if multi_gpu:
        model = nn.DataParallel(model)
    if "cuda" in device:
        torch.cuda.set_device(device)
        model = model.cuda()

    print("[INFO] Model initialized!")

    # Load saved model state into freshly initialized model
    utils.load_checkpoint(model_filepath, model)
    return model


# Do inference with loaded model on data of choice
def inference(
    model, data, labels=None, perform_fn=None, move_channel=True, output_np=True
):
    """Performs inference with a model on the given data.

    Args:
        model (torch.nn.Module): A PyTorch model as the nn.Module class.
        data (torch.tensor): PyTorch tensor of data with structure: [batch, channel, height, width].
        labels (torch.tensor, optional): PyTorch tensor of labels. Defaults to None.
        perform_fn (function, optional): Function that calculates performance. Defaults to None.
        move_channel (bool, optional): If True, moves channel from 2nd to 4th dimension. Defaults to True.
        output_np (bool, optional): If True, converts output tensor to np.ndarray. Defaults to True.

    Returns:
        tensor/tuple(tensor, list): Tensor of predictions and optionally performance as list of floats.
    """
    # Make predictions
    model = model.eval()
    with torch.inference_mode():
        logit_preds = model(data)
        preds = torch.sigmoid(logit_preds)

    # Move channel from 2nd to 4th dimension if desired
    if move_channel:
        preds = preds.permute([0, 2, 3, 1])

    # Push to CPU and convert to np.ndarray if desired
    if output_np:
        preds = preds.detach().cpu().numpy()

    # Calculate performance if desired
    if (labels is not None) and (perform_fn is not None):
        performs = []
        for logit_pred, label in zip(logit_preds, labels):
            batch_pred = logit_pred.unsqueeze(0)
            batch_label = label.reshape(batch_pred.shape)
            perform = perform_fn(batch_pred, batch_label)
            performs.append(perform)
        return preds, performs

    else:
        return preds


# Plot multiple images on one row
def show(imgs, save_path=None, save_dpi=300):
    """Plots multiple images on a row.

    Adapted from:
        https://pytorch.org/vision/main/auto_examples/others/plot_visualization_utils.html

    Args:
        imgs (list, or convertible to list): List of images.
        save_path (str): Path to save figure to. None to not save figure.
        save_dpi (int): Dpi with which to save figure if desired.
    """
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    for pos in ["right", "top", "bottom", "left"]:
        plt.gca().spines[pos].set_visible(False)
    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0, dpi=save_dpi)
    plt.show()


# Define dir with images
class LettuceSegNoLabelDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        """Creates a PyTorch Dataset class of image data for inference.

        Loads data by providing the image and the corresponding filename.
        The filename is given alongside the image to allow for automated naming of output files.

        Args:
            img_dir (str): Path to directory containing the input images.
            transform (albumentations.Compose, optional): Transformations for data aug. Defaults to None.
        """
        self.transform = transform

        # List all image filenames
        self.img_names = os.listdir(img_dir)

        # Create lists of filepath for images and masks
        self.img_paths = []
        for img_name in self.img_names:
            img_path = os.path.join(img_dir, img_name)
            self.img_paths.append(img_path)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img = np.array(Image.open(self.img_paths[index]))

        if img.shape[2] == 4:
            img = img[:, :, :3]

        if self.transform is not None:
            augmentations = self.transform(image=img)
            img = augmentations["image"]

        return img, self.img_names[index]


def main():
    # Set globals and directories
    DEVICE = "cuda:0"
    MULTI_GPU = True
    PERFORM_FN = None
    RGB_ALPHA = True

    img_dir = "/lustre/BIF/nobackup/to001/thesis_MBF/data/people"
    label_dir = None
    target_dir = "/lustre/BIF/nobackup/to001/thesis_MBF/data/people/people_tb_masks"
    savename_appendix = "_tb_UnetMit-b3_lr1e-4_b32_Ldice_ep100_tb_mask.png"
    perform_save_name = None

    transforms = A.Compose([A.Resize(height=480, width=480), ToTensorV2()])

    model_dir = "/lustre/BIF/nobackup/to001/thesis_MBF/output"
    model_name = "tb_UnetMit-b3_lr1e-4_b32_Ldice_ep100.pth.tar"

    # From directory inference
    # With performance tracking
    if (PERFORM_FN is not None) and (label_dir is not None):
        dataset = data_setup.LettuceSegDataset(
            img_dir=img_dir,
            label_dir=label_dir,
            is_train=None,
            train_frac=1,
            transform=transforms,
            give_name=True,
        )
        loader = DataLoader(
            dataset,
            batch_size=8,
            num_workers=2,
            pin_memory=True,
            shuffle=False,
        )

        # Load model
        full_model_path = os.path.join(
            model_dir, model_name.split(os.extsep)[0], model_name
        )
        model = load_model(full_model_path, device=DEVICE, multi_gpu=MULTI_GPU)

        # Track performance
        target_dir_path = Path(target_dir)
        target_dir_path.mkdir(parents=True, exist_ok=True)
        perform_path = os.path.join(target_dir, perform_save_name)
        with open(perform_path, "w") as perform_tsv:
            # Make predictions
            for batch in tqdm(loader, desc="Batches"):
                input_imgs, labels, filenames = batch

                # Move data and labels to device
                if "cuda" in DEVICE:
                    torch.cuda.set_device(DEVICE)
                    input_imgs = input_imgs.cuda()
                    labels = labels.cuda()

                # Get output from model
                output_masks, performs = inference(
                    model,
                    input_imgs.float(),
                    labels.float(),
                    perform_fn=PERFORM_FN,
                    move_channel=False,
                    output_np=False,
                )
                output_masks = output_masks.round().bool()

                for output_mask, input_img, filename, perform in zip(
                    output_masks, input_imgs, filenames, performs
                ):
                    # Adjust mask before saving
                    if RGB_ALPHA:
                        # Apply predicted mask on input image
                        masked_img = draw_segmentation_masks(
                            input_img, ~output_mask, alpha=0.8
                        )
                        masked_img = masked_img.detach()
                        masked_img = F.to_pil_image(masked_img)
                        masked_img = np.asarray(masked_img)
                    else:
                        # Output mask as binary image
                        masked_img = output_mask[0, :, :]
                        masked_img = masked_img.detach().cpu().numpy()
                        masked_img = util.img_as_ubyte(masked_img)

                    # Save image
                    new_name = f"{filename.split(os.extsep)[0]}{savename_appendix}"
                    utils.save_img(masked_img, target_dir=target_dir, filename=new_name)

                    # Write performance to tsv file
                    new_line = f"{new_name}\t{perform}\n"
                    perform_tsv.write(new_line)
        print(f"Performance has been saved to {perform_path}!")

    # Without performance tracking
    else:
        dataset = LettuceSegNoLabelDataset(img_dir=img_dir, transform=transforms)
        loader = DataLoader(
            dataset,
            batch_size=8,
            num_workers=2,
            pin_memory=True,
            shuffle=False,
        )

        # Load model
        full_model_path = os.path.join(
            model_dir, model_name.split(os.extsep)[0], model_name
        )
        model = load_model(full_model_path, device=DEVICE, multi_gpu=MULTI_GPU)

        # Make predictions
        for batch in tqdm(loader, desc="Batches"):
            input_imgs, filenames = batch

            # Move data to device
            if "cuda" in DEVICE:
                torch.cuda.set_device(DEVICE)
                input_imgs = input_imgs.cuda()

            # Get output from model
            output_masks = inference(
                model,
                input_imgs.float(),
                move_channel=False,
                output_np=False,
            )
            output_masks = output_masks.round().bool()

            # Create target directory to save
            target_dir_path = Path(target_dir)
            target_dir_path.mkdir(parents=True, exist_ok=True)

            for output_mask, input_img, filename in zip(
                output_masks, input_imgs, filenames
            ):
                # Adjust mask before saving
                if RGB_ALPHA:
                    # Apply predicted mask on input image
                    masked_img = draw_segmentation_masks(
                        input_img, ~output_mask, alpha=0.9
                    )
                    masked_img = masked_img.detach()
                    masked_img = F.to_pil_image(masked_img)
                    masked_img = np.asarray(masked_img)
                else:
                    # Output mask as binary image
                    masked_img = output_mask[0, :, :]
                    masked_img = masked_img.cpu().numpy()
                    masked_img = util.img_as_ubyte(masked_img)

                # Save image
                new_name = f"{filename.split(os.extsep)[0]}{savename_appendix}"
                utils.save_img(masked_img, target_dir=target_dir, filename=new_name)


if __name__ == "__main__":
    main()
