#!/usr/bin/env python3
"""
Analyzes different trained models using loss & performance plots and inference.
"""

# Import statements
import os
import torch
import albumentations as A
import cv2
import numpy as np
import matplotlib.pyplot as plt
from albumentations.pytorch import ToTensorV2

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


def main():
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

    # Load model
    output_dir = "/lustre/BIF/nobackup/to001/thesis_MBF/output"
    model_path = "PANResNest101_lr1e-4_b32_Ldicebce_ep40/PANResNest101_lr1e-4_b32_Ldicebce_ep40.pth.tar"
    full_model_path = os.path.join(output_dir, model_path)
    model = load_model(full_model_path)

    # Load data and labels
    data, labels = next(iter(test_loader))
    data = data.permute([0, 2, 3, 1])

    # Plot images
    num_imgs = 4
    offset = 17
    fig, axes = plt.subplots(1, num_imgs)
    for i in range(num_imgs):
        img = data[i + offset]
        axes[i].imshow(img)
        axes[i].axis("off")
    fig.tight_layout()
    plt.savefig(
        "/lustre/BIF/nobackup/to001/thesis_MBF/output/inference_org.png", dpi=300
    )
    plt.show()

    # Plot ground truth
    fig, axes = plt.subplots(1, num_imgs)
    for i in range(num_imgs):
        img = data[i + offset] * np.repeat(
            labels[i + offset][:, :, np.newaxis], repeats=3, axis=2
        )
        axes[i].imshow(img)
        axes[i].axis("off")
    fig.tight_layout()
    plt.savefig(
        "/lustre/BIF/nobackup/to001/thesis_MBF/output/inference_gt.png", dpi=300
    )
    plt.show()

    # Plot predictions
    fig, axes = plt.subplots(1, num_imgs)
    input_imgs = (
        data[offset : (num_imgs + offset)].permute([0, 3, 1, 2]).to("cuda").float()
    )
    output_masks = (
        torch.sigmoid(model(input_imgs))
        .permute([0, 2, 3, 1])
        .detach()
        .cpu()
        .numpy()
        .round(),
    )
    for i in range(num_imgs):
        img = data[i + offset] * np.repeat(
            output_masks[0][i], repeats=3, axis=2
        ).astype(np.uint8)
        axes[i].imshow(img)
        axes[i].axis("off")
    fig.tight_layout()
    plt.savefig(
        "/lustre/BIF/nobackup/to001/thesis_MBF/output/inference_PAN.png",
        dpi=300,
    )
    plt.show()


if __name__ == "__main__":
    main()
