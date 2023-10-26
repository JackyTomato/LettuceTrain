#!/usr/bin/env python3
"""
Trains a PyTorch model according to the user's configuration in "config.json".

For a complete explanation of the framework and how to use it, check out:
    https://github.com/JackyTomato/LettuceTrain

Currently only supports image classifiers.

The script should be run from the /scripts/ directory
to make sure all file paths are correct. If you would still like to run
from a different working directory, adjust the variable 'new_cwd' in
import statements to your /scripts/ file path.

The script uses a parsed config.json file from config_parser in
the /scripts/ directory to obtain all the necessary information.
The config includes seed, hyperparameters, device, data loading,
model and save settings.

Example of config.json with brief explanations as comments:
    "SEED": 42, # int, random seed used for reproducbility
    "LEARNING_RATE": # 1e-2, float, learning rate
    "NUM_EPOCHS": 4, # int, number of epochs
    "OPTIMIZER": "torch.optim.AdamW", # torch.optim, PyTorch optimizer class
    "SCALER": "torch.cuda.amp.GradScaler()", # torch.cuda.amp, PyTorch scaler class
    "LOSS_FN": "nn.CrossEntropyLoss()", # torch.nn, PyTorch loss class
    "PERFORMANCE_FN": "utils.class_accuracy", # function, a performance metric function
    "DEVICE": "cuda", # str, device to train "cuda" (GPU) or "cpu"
    "NUM_WORKERS": 4, # int, number of worker processes for data loading
    "PIN_MEMORY": "True", # bool, if True speeds up data transfer from CPU to GPU
    "DATA_CLASS": "data_setup.LettuceDataset", torch.utils.data.Dataset, PyTorch dataset class
    "IMG_DIR": "data/img", # str, filepath of dir containing the imaging data
    "LABEL_DIR": "data/label", # str, filepath of dir containing image labels
    "TRAIN_FRAC": 0.75, # float, fraction of dataset to use for training
    "TRANSFORMS": [
        "A.Resize(height=512, width=512)",
        "ToTensorV2()"
    ], # list, list of albumentations or torchvision transforms for data augmentation
    "BATCH_SIZE": 2048, # int, size of batches to load from data
    "MODEL_TYPE": "model_builder.TipburnClassifier", # torch.nn.Module, PyTorch Module class
    "N_CLASSES": 10, # int, number of output classes to predict
    "N_CHANNELS": 1, # int, number of input channels of data
    "BB_NAME": "wide_resnet50_2", # torchvision.models, model from torchvision.models for backbone
    "BB_WEIGHTS": "IMAGENET1K_V2", # str, name weights for pretraining, "None" for no pretraining
    "BB_FREEZE": "True", # bool, if True freezes weights of backbone
    "CHECKPOINT_FREQ": 2, # int, the model will be saved every number of epochs equal to this value
    "SAVE_MODEL_DIR": "/output/model1", # str, filepath to which to save the model to
    "SAVE_MODEL_NAME": "test_model1.pth.tar", # str (.pt, .pth, .pt.tar, .pth.tar), filename to save model
    "LOAD_MODEL": "False", # bool, if True loads saved model before training
    "LOAD_MODEL_PATH": "output/model1/test_model1.pth.tar" # str, filepath of saved model to load
}

TODO:)
    - Allow user to specify config locations from terminal with arg
    - Make torchvision summary extract input image size from data
    - Create support for more than just classifiers (separate scripts?)
"""

# Import statements
import os
import torch
import numpy as np
import random
from tqdm import tqdm

# Import supporting modules
if "scripts" not in os.getcwd():
    # Change wd to scripts if cwd is not scripts
    new_cwd = "/lustre/BIF/nobackup/to001/thesis_MBF/scripts"
    print(f"[INFO] Changing working directory to {new_cwd}")
    os.chdir(new_cwd)
import data_setup, engine, model_builder, utils
import config_parser as cp

print("[INFO] Loading config.json was succesful!")


def main():
    # Set seeds for reproducibility
    torch.manual_seed(cp.SEED)
    random.seed(cp.SEED)
    np.random.seed(cp.SEED)

    # Create DataLoaders with help from data_setup.py
    train_loader, test_loader = data_setup.get_loaders(
        dataset=cp.DATASET,
        img_dir=cp.IMG_DIR,
        label_dir=cp.LABEL_DIR,
        train_frac=cp.TRAIN_FRAC,
        train_augs=cp.TRAIN_TRANSFORMS,
        test_augs=cp.TEST_TRANSFORMS,
        batch_size=cp.BATCH_SIZE,
        num_workers=cp.NUM_WORKERS,
        pin_memory=cp.PIN_MEMORY,
    )
    print("[INFO] Data succesfully loaded!")

    # Create model with help from model_builder.py and send to device
    model = cp.MODEL_TYPE(
        model_name=cp.MODEL_NAME,
        encoder_name=cp.ENCODER_NAME,
        encoder_weights=cp.ENCODER_WEIGHTS,
        n_channels=cp.N_CHANNELS,
        n_classes=cp.N_CLASSES,
        decoder_attention=cp.DECODER_ATTENTION,
        encoder_freeze=cp.ENCODER_FREEZE,
    ).to(cp.DEVICE)
    print("[INFO] Model initialized!")

    # Start training with help from engine.py
    # Load model if requested
    if cp.LOAD_MODEL == True:
        utils.load_checkpoint(checkpoint=cp.LOAD_MODEL_PATH, model=model)

    # Prepare optimizer
    cp.OPTIMIZER = cp.OPTIMIZER(params=model.parameters(), lr=cp.LEARNING_RATE)

    # Create empty results dictionary for loss and performance during training loop
    results = {
        "epoch": [],
        "train_loss": [],
        "train_perform": [],
        "test_loss": [],
        "test_perform": [],
    }

    # Setup tqdm loop for progress bar over epochs
    epoch_loop = tqdm(range(cp.NUM_EPOCHS), desc="Epochs")

    # Training loop for a number of epochs
    for epoch in epoch_loop:
        train_loss, train_perform = engine.train_step(
            model=model,
            dataloader=train_loader,
            loss_fn=cp.LOSS_FN,
            performance_fn=cp.PERFORMANCE_FN,
            optimizer=cp.OPTIMIZER,
            scaler=cp.SCALER,
            device=cp.DEVICE,
        )
        test_loss, test_perform = engine.test_step(
            model=model,
            dataloader=test_loader,
            loss_fn=cp.LOSS_FN,
            performance_fn=cp.PERFORMANCE_FN,
            device=cp.DEVICE,
        )

        # Checkpoint model at a given frequency if requested
        if cp.CHECKPOINT_FREQ is not None:
            if (epoch + 1) % cp.CHECKPOINT_FREQ == 0:  # Current epoch is epoch + 1
                checkpoint = {
                    "state_dict": model.state_dict(),
                    "optimizer": cp.OPTIMIZER.state_dict(),
                }
                utils.save_checkpoint(
                    state=checkpoint,
                    target_dir=cp.SAVE_MODEL_DIR,
                    model_name=cp.SAVE_MODEL_NAME.split(os.extsep)[0],
                )

        # Print out epoch number, loss and performance for this epoch
        print(
            f"Epoch: {epoch + 1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_perform: {train_perform:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_perform: {test_perform:.4f}"
        )

        # Update results dictionary
        results["epoch"].append(epoch + 1)
        results["train_loss"].append(train_loss)
        results["train_perform"].append(train_perform)
        results["test_loss"].append(test_loss)
        results["test_perform"].append(test_perform)

    # Save the model with help from utils.py
    if cp.CHECKPOINT_FREQ is None:
        final_state = {
            "state_dict": model.state_dict(),
            "optimizer": cp.OPTIMIZER.state_dict(),
        }
        utils.save_checkpoint(
            state=final_state,
            target_dir=cp.SAVE_MODEL_DIR,
            model_name=cp.SAVE_MODEL_NAME.split(os.extsep)[0],
        )
    elif (
        cp.cp.NUM_EPOCHS % cp.CHECKPOINT_FREQ != 0
    ):  # Don't save when final epoch was checkpoint
        final_state = {
            "state_dict": model.state_dict(),
            "optimizer": cp.OPTIMIZER.state_dict(),
        }
        utils.save_checkpoint(
            state=final_state,
            target_dir=cp.SAVE_MODEL_DIR,
            model_name=cp.SAVE_MODEL_NAME.split(os.extsep)[0],
        )

    # Save loss and performance during training
    utils.save_train_results(
        dict_results=results,
        target_dir=cp.SAVE_MODEL_DIR,
        filename=f"results_{cp.SAVE_MODEL_NAME.split(os.extsep)[0]}.tsv",
    )

    # Save a torchinfo summary of the network
    utils.save_network_summary(
        model=model,
        target_dir=cp.SAVE_MODEL_DIR,
        filename=f"summary_{cp.SAVE_MODEL_NAME.split(os.extsep)[0]}.txt",
        n_channels=cp.N_CHANNELS,
    )

    # Save the config
    utils.save_config(
        target_dir=cp.SAVE_MODEL_DIR,
        filename=f"config_{cp.SAVE_MODEL_NAME.split(os.extsep)[0]}.json",
    )


if __name__ == "__main__":
    main()
