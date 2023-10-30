#!/usr/bin/env python3
"""
Analyzes different trained models using loss & performance plots and inference.

This script was ran in similar fashion to Rstudio, line-by-line
for quick on-the-go predictions of different models on different images.
Thus, the script has not been designed to be run in its entirety.
"""

# Import statements
import os
import torch
import torchvision.transforms.functional as F
import albumentations as A
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import make_grid, draw_segmentation_masks, save_image
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from pathlib import Path

# Import supporting modules
import data_setup, model_builder, utils


# Load model from saved model filename
def load_model(model_filepath, device="cuda"):
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
    ).to(device)
    print("[INFO] Model initialized!")

    # Load saved model state into freshly initialized model
    utils.load_checkpoint(model_filepath, model)
    return model


# Do inference with loaded model on data of choice
def inference(model, data, move_channel=True, output_np=True):
    """Performs inference with a model on the given data.

    Args:
        model (torch.nn.Module): A PyTorch model as the nn.Module class.
        data (torch.tensor): PyTorch tensor of data with structure: [batch, channel, height, width].
        move_channel (bool, optional): If True, moves channel from 2nd to 4th dimension. Defaults to True.
        output_np (bool, optional): If True, converts output tensor to np.ndarray. Defaults to True.

    Returns:
        _type_: _description_
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
class InferDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        """Creates a PyTorch Dataset class of image data for inference.

        oads data by providing the image and the corresponding filename.
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
    # Mini-batch inference
    # Load data
    train_transforms = A.Compose(
        [
            A.Resize(height=480, width=480),
            A.Rotate(limit=180, border_mode=cv2.BORDER_CONSTANT, p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=0.5),
            A.ColorJitter(
                brightness=0.05, contrast=0.05, saturation=0.05, hue=0.05, p=0.5
            ),
            ToTensorV2(),
        ]
    )
    test_transforms = A.Compose([A.Resize(height=480, width=480), ToTensorV2()])
    train_loader, test_loader = data_setup.get_loaders(
        dataset=data_setup.LettuceSegDataset,
        img_dir="/lustre/BIF/nobackup/to001/thesis_MBF/data/TrainTest/rgb_crops",
        label_dir="/lustre/BIF/nobackup/to001/thesis_MBF/data/TrainTest/rgb_masks",
        train_frac=0.75,
        train_augs=train_transforms,
        test_augs=test_transforms,
        batch_size=32,
        num_workers=8,
        pin_memory=True,
    )

    # Load data and labels
    data, labels = next(iter(test_loader))
    imgs = data[17:21]
    gt_masks = labels[17:21]

    # Plot original images
    org_grid = make_grid(imgs, padding=0)
    show(
        org_grid,
        save_path="/lustre/BIF/nobackup/to001/thesis_MBF/inference/out/inference_org.png",
    )

    # Plot images masked with ground truth
    gt_imgs = [
        draw_segmentation_masks(img.to(torch.uint8), ~mask.bool())
        for img, mask in zip(imgs, gt_masks)
    ]

    gt_grid = make_grid(gt_imgs, padding=0)
    show(
        gt_grid,
        save_path="/lustre/BIF/nobackup/to001/thesis_MBF/inference/out/inference_gt.png",
    )

    # Plot images masked with predictions
    output_dir = "/lustre/BIF/nobackup/to001/thesis_MBF/output"
    model_name = "PANRes2Net50-14_lr1e-4_b32_Ldicebce_ep100.pth.tar"
    full_model_path = os.path.join(
        output_dir, model_name.split(os.extsep)[0], model_name
    )
    device = "cuda"
    model = load_model(full_model_path, device=device)

    output_masks = inference(
        model,
        imgs.float().to(device),
        move_channel=False,
        output_np=False,
    ).round()

    masked_imgs = [
        draw_segmentation_masks(img.to(torch.uint8), ~mask.bool())
        for img, mask in zip(imgs, output_masks)
    ]

    pred_grid = make_grid(masked_imgs, padding=0)
    show(
        pred_grid,
        save_path="/lustre/BIF/nobackup/to001/thesis_MBF/inference/out/inference_PANRes2Net.png",
    )

    # From directory inference
    # Define dir with images
    img_dir = "/lustre/BIF/nobackup/to001/thesis_MBF/inference/in/chris_1tp"
    transforms = A.Compose([A.Resize(height=480, width=480), ToTensorV2()])

    # Load data
    dataset = InferDataset(img_dir=img_dir, transform=transforms)
    loader = DataLoader(
        dataset,
        batch_size=16,
        num_workers=4,
        pin_memory=True,
        shuffle=False,
    )

    # Make predictions
    for input_imgs, filenames in tqdm(loader, desc="Batches"):
        output_masks = inference(
            model,
            input_imgs.float().to(device),
            move_channel=False,
            output_np=False,
        )
        output_masks = output_masks.round().bool()
        for output_mask, input_img, filename in zip(
            output_masks, input_imgs, filenames
        ):
            # Create target directory to save
            target_dir = "/lustre/BIF/nobackup/to001/thesis_MBF/inference/out/chris_1tp"
            target_dir_path = Path(target_dir)
            target_dir_path.mkdir(parents=True, exist_ok=True)

            # Apply predicted mask on img
            masked_img = draw_segmentation_masks(input_img, ~output_mask)
            masked_img = masked_img.float()
            masked_img = (masked_img - masked_img.min()) / (
                masked_img.max() - masked_img.min()
            )

            # Save image
            save_filepath = os.path.join(
                target_dir, f"{filename.split(os.extsep)[0]}_PANmask.png"
            )
            save_image(masked_img, fp=save_filepath)


if __name__ == "__main__":
    main()
