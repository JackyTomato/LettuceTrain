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
    "DEVICE": "cuda", # str, device to train "cuda" (GPU) or "cpu"
    "MULTI_GPU": "True", # bool, if True splits data over multiple GPU for training
    "NUM_WORKERS": 4, # int, number of worker processes for data loading
    "PIN_MEMORY": "True", # bool, if True speeds up data transfer from CPU to GPU
    "LEARNING_RATE": # 1e-2, float, learning rate
    "NUM_EPOCHS": 4, # int, number of epochs
    "OPTIMIZER": "torch.optim.AdamW", # torch.optim, PyTorch optimizer class
    "SCALER": "torch.cuda.amp.GradScaler()", # torch.cuda.amp, PyTorch scaler class
    "LOSS_FN": "torch.nn.CrossEntropyLoss()", # torch.nn, PyTorch loss class
    "PERFORMANCE_FN": "utils.class_accuracy", # function, a performance metric function
    "DATA_CLASS": "data_setup.LettuceSegDataset", torch.utils.data.Dataset, PyTorch dataset class
    "IMG_DIR": "data/img", # str, filepath of dir containing the imaging data
    "LABEL_DIR": "data/label", # str, filepath of dir containing image labels
    "TRAIN_FRAC": 0.75, # float, fraction of dataset to use for training or [0.4, 0.8] # list of floats, for trainset size testing
    "KFOLD" : 4, # int, K for K-fold cross validation
    "TRANSFORMS": [
        "A.Resize(height=512, width=512)",
        "ToTensorV2()"
    ], # list, list of albumentations or torchvision transforms for data augmentation
    "TRAIN_TRANSFORMS": [
        "A.Resize(height=480, width=480)",
        "A.Rotate(limit=180, border_mode=cv2.BORDER_CONSTANT, p=0.5)",
        "A.HorizontalFlip(p=0.5)",
        "A.VerticalFlip(p=0.5)",
        "A.RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=0.5)",
        "A.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.05, p=0.5)",
        "ToTensorV2()"
    ], # list, list of albumentations or torchvision transforms for train data augmentation
    "TEST_TRANSFORMS": [
        "A.Resize(height=480, width=480)",
        "ToTensorV2()"
    ], # list, list of albumentations or torchvision transforms for test data augmentation
    "BATCH_SIZE": 2048, # int, size of batches to load from data
    "MODEL_TYPE": "model_builder.TipburnClassifier", # torch.nn.Module, PyTorch Module class
    "MODEL_NAME": "DeepLabV3Plus", # torchvision.models or segmentation_models_pytorch, model name
    "ENCODER_NAME": "tu-resnest101e", # str, segmentation_models_pytorch encoder name
    "ENCODER_WEIGHTS": "imagenet", # str, name of pretrained weights
    "N_CHANNELS": 3, # int, number of input channels of data
    "N_CLASSES": 1, # int, number of output classes to predict
    "DECODER_ATTENTION": "None", # bool, set decoder attenton type for Unet in segemtnation_models_pytorch
    "ENCODER_FREEZE": "False", # bool, freeze parameters in encoder
    "CHECKPOINT_FREQ": 2, # int, the model will be saved every number of epochs equal to this value
    "SAVE_MODEL_DIR": "/output/model1", # str, filepath to which to save the model to
    "SAVE_MODEL_NAME": "test_model1.pth.tar", # str (.pt, .pth, .pt.tar, .pth.tar), filename to save model
    "LOAD_MODEL": "False", # bool, if True loads saved model before training
    "LOAD_MODEL_PATH": "output/model1/test_model1.pth.tar" # str, filepath of saved model to load
}

TODO:
    - Allow user to specify config locations from terminal with arg
    - Make torchvision summary extract input image size from data
    - Make tensorboard script to study different runs
    - Allow config to specify multi-GPU training
"""

# Import statements
import os
import torch
import torch.nn as nn
import numpy as np
import random
import gc
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

    # Normal training, no K-fold cross validation
    if cp.KFOLD is None:
        # Setup tqdm loop for progress bar over trainset size testing runs with multiple training fractions
        if len(cp.TRAIN_FRAC) > 1:
            train_frac_loop = tqdm(cp.TRAIN_FRAC, desc="Trainset size testing runs")
        else:
            train_frac_loop = cp.TRAIN_FRAC
        for run, train_frac in enumerate(train_frac_loop):
            # Add train fraction to model save name when performing trainset size testing
            # Also announce trainset size testing
            if len(cp.TRAIN_FRAC) > 1:
                split_filename = cp.SAVE_MODEL_NAME.split(os.extsep, 1)
                str_train_frac = str(train_frac).replace(".", "")
                save_model_name = (
                    f"{split_filename[0]}_frac{str_train_frac}.{split_filename[1]}"
                )
                if run == 0:
                    print(
                        f"[INFO] Performing trainset size testing with {len(cp.TRAIN_FRAC)} training fractions!"
                    )
            else:
                save_model_name = cp.SAVE_MODEL_NAME
            # Create DataLoaders with help from data_setup.py
            loaders = data_setup.get_loaders(
                dataset=cp.DATASET,
                img_dir=cp.IMG_DIR,
                label_dir=cp.LABEL_DIR,
                fm_dir=cp.FM_DIR,
                fvfm_dir=cp.FVFM_DIR,
                train_frac=train_frac,
                kfold=cp.KFOLD,
                train_augs=cp.TRAIN_TRANSFORMS,
                test_augs=cp.TEST_TRANSFORMS,
                batch_size=cp.BATCH_SIZE,
                num_workers=cp.NUM_WORKERS,
                pin_memory=cp.PIN_MEMORY,
                seed=cp.SEED,
            )
            print("[INFO] Data succesfully loaded!")

            # Clean up old objects and free up GPU memory if not first run
            if run > 0:
                del model, optimizer, results
                gc.collect()
                if cp.DEVICE == "cuda":
                    torch.cuda.empty_cache()
                print(
                    f"[INFO] Re-initializing model, optimizer and results logger for run {run + 1}, with training fraction: {train_frac}!"
                )

            # (Re-)initialize model for next fold with help from model_builder.py and send to device
            model = cp.MODEL_TYPE(
                model_name=cp.MODEL_NAME,
                encoder_name=cp.ENCODER_NAME,
                encoder_weights=cp.ENCODER_WEIGHTS,
                n_channels=cp.N_CHANNELS,
                n_classes=cp.N_CLASSES,
                decoder_attention=cp.DECODER_ATTENTION,
                encoder_freeze=cp.ENCODER_FREEZE,
                fusion=cp.FUSION,
                n_channels_med1=cp.N_CHANNELS_MED1,
                n_channels_med2=cp.N_CHANNELS_MED2,
            )
            if cp.MULTI_GPU:
                model = nn.DataParallel(model)
            model = model.to(cp.DEVICE)

            # Load model if requested
            if cp.LOAD_MODEL == True:
                utils.load_checkpoint(checkpoint=cp.LOAD_MODEL_PATH, model=model)

            # Prepare optimizer
            optimizer = cp.OPTIMIZER(params=model.parameters(), lr=cp.LEARNING_RATE)

            # Create empty results dictionary for loss and performance during training loop
            results = {
                "epoch": [],
                "train_loss": [],
                "train_perform": [],
                "test_loss": [],
                "test_perform": [],
            }

            # Prepare loaders for loop over epochs without K-fold CV
            train_loader, test_loader = loaders

            # Setup tqdm loop for progress bar over epochs
            epoch_loop = tqdm(range(cp.NUM_EPOCHS), desc="Epochs")
            for epoch in epoch_loop:
                train_loss, train_perform = engine.train_step(
                    model=model,
                    dataloader=train_loader,
                    loss_fn=cp.LOSS_FN,
                    performance_fn=cp.PERFORMANCE_FN,
                    optimizer=optimizer,
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
                    if (
                        epoch + 1
                    ) % cp.CHECKPOINT_FREQ == 0:  # Current epoch is epoch + 1
                        checkpoint = {
                            "state_dict": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                        }
                        utils.save_checkpoint(
                            state=checkpoint,
                            target_dir=cp.SAVE_MODEL_DIR,
                            model_name=save_model_name,
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
                    "optimizer": optimizer.state_dict(),
                }
                utils.save_checkpoint(
                    state=final_state,
                    target_dir=cp.SAVE_MODEL_DIR,
                    model_name=save_model_name,
                )
            elif (
                cp.NUM_EPOCHS % cp.CHECKPOINT_FREQ != 0
            ):  # Don't save when final epoch was checkpoint
                final_state = {
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }
                utils.save_checkpoint(
                    state=final_state,
                    target_dir=cp.SAVE_MODEL_DIR,
                    model_name=save_model_name,
                )

            # Save loss and performance during training
            utils.save_train_results(
                dict_results=results,
                target_dir=cp.SAVE_MODEL_DIR,
                filename=f"results_{save_model_name.split(os.extsep)[0]}.tsv",
            )

            # Save a torchinfo summary of the network
            utils.save_network_summary(
                model=model,
                target_dir=cp.SAVE_MODEL_DIR,
                filename=f"summary_{save_model_name.split(os.extsep)[0]}.txt",
                n_channels=cp.N_CHANNELS,
            )

            # Save the config
            utils.save_config(
                target_dir=cp.SAVE_MODEL_DIR,
                filename=f"config_{save_model_name.split(os.extsep)[0]}.json",
            )

            if len(cp.TRAIN_FRAC) > 1:
                print(
                    f"[INFO] Run {run + 1}, with training fraction {train_frac} finished!"
                )

    # Perform K-fold cross validaton
    else:
        print(f"[INFO] Performing {cp.KFOLD}-fold cross-validation")

        # Create DataLoaders with help from data_setup.py
        loaders = data_setup.get_loaders(
            dataset=cp.DATASET,
            img_dir=cp.IMG_DIR,
            label_dir=cp.LABEL_DIR,
            fm_dir=cp.FM_DIR,
            fvfm_dir=cp.FVFM_DIR,
            train_frac=cp.TRAIN_FRAC,
            kfold=cp.KFOLD,
            train_augs=cp.TRAIN_TRANSFORMS,
            test_augs=cp.TEST_TRANSFORMS,
            batch_size=cp.BATCH_SIZE,
            num_workers=cp.NUM_WORKERS,
            pin_memory=cp.PIN_MEMORY,
            seed=cp.SEED,
        )
        print("[INFO] Data succesfully loaded!")

        # Setup tqdm loop for progress bar over K-folds
        kfold_loop = tqdm(loaders, desc="Cross Validation Folds")
        for fold, (train_loader, test_loader) in enumerate(kfold_loop):
            # Clean up old objects and free up GPU memory if not first fold
            if fold > 0:
                del model, optimizer, results
                gc.collect()
                if cp.DEVICE == "cuda":
                    torch.cuda.empty_cache()
                print(
                    f"[INFO] Re-initializing model, optimizer and results logger for fold {fold + 2}!"
                )

            # (Re-)initialize model for next fold with help from model_builder.py and send to device
            model = cp.MODEL_TYPE(
                model_name=cp.MODEL_NAME,
                encoder_name=cp.ENCODER_NAME,
                encoder_weights=cp.ENCODER_WEIGHTS,
                n_channels=cp.N_CHANNELS,
                n_classes=cp.N_CLASSES,
                decoder_attention=cp.DECODER_ATTENTION,
                encoder_freeze=cp.ENCODER_FREEZE,
                fusion=cp.FUSION,
                n_channels_med1=cp.N_CHANNELS_MED1,
                n_channels_med2=cp.N_CHANNELS_MED2,
            )
            if cp.MULTI_GPU:
                model = nn.DataParallel(model)
            model = model.to(cp.DEVICE)

            # Load model if requested
            if cp.LOAD_MODEL == True:
                utils.load_checkpoint(checkpoint=cp.LOAD_MODEL_PATH, model=model)

            # Prepare optimizer
            optimizer = cp.OPTIMIZER(params=model.parameters(), lr=cp.LEARNING_RATE)

            # Create empty results dictionary for loss and performance during training loop
            results = {
                "epoch": [],
                "train_loss": [],
                "train_perform": [],
                "test_loss": [],
                "test_perform": [],
            }

            # Create model save name
            model_name_split = cp.SAVE_MODEL_NAME.split(os.extsep, 1)
            model_new_name = (
                f"{model_name_split[0]}_fold{fold + 1}.{model_name_split[1]}"
            )

            # Setup tqdm loop for progress bar over epochs
            epoch_loop = tqdm(range(cp.NUM_EPOCHS), desc="Epochs")
            for epoch in epoch_loop:
                train_loss, train_perform = engine.train_step(
                    model=model,
                    dataloader=train_loader,
                    loss_fn=cp.LOSS_FN,
                    performance_fn=cp.PERFORMANCE_FN,
                    optimizer=optimizer,
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
                    if (
                        epoch + 1
                    ) % cp.CHECKPOINT_FREQ == 0:  # Current epoch is epoch + 1
                        checkpoint = {
                            "state_dict": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                        }
                        utils.save_checkpoint(
                            state=checkpoint,
                            target_dir=cp.SAVE_MODEL_DIR,
                            model_name=model_new_name,
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
                    "optimizer": optimizer.state_dict(),
                }
                utils.save_checkpoint(
                    state=final_state,
                    target_dir=cp.SAVE_MODEL_DIR,
                    model_name=model_new_name,
                )
            elif (
                cp.NUM_EPOCHS % cp.CHECKPOINT_FREQ != 0
            ):  # Don't save when final epoch was checkpoint
                final_state = {
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }
                utils.save_checkpoint(
                    state=final_state,
                    target_dir=cp.SAVE_MODEL_DIR,
                    model_name=model_new_name,
                )

            # Save loss and performance during training
            utils.save_train_results(
                dict_results=results,
                target_dir=cp.SAVE_MODEL_DIR,
                filename=f"results_{cp.SAVE_MODEL_NAME.split(os.extsep)[0]}_fold{fold + 1}.tsv",
            )

            # Save a torchinfo summary of the network
            utils.save_network_summary(
                model=model,
                target_dir=cp.SAVE_MODEL_DIR,
                filename=f"summary_{cp.SAVE_MODEL_NAME.split(os.extsep)[0]}_fold{fold + 1}.txt",
                n_channels=cp.N_CHANNELS,
            )

            # Save the config
            utils.save_config(
                target_dir=cp.SAVE_MODEL_DIR,
                filename=f"config_{cp.SAVE_MODEL_NAME.split(os.extsep)[0]}_fold{fold + 1}.json",
            )

            print(f"[INFO] Fold {fold + 1} finished!")


if __name__ == "__main__":
    main()
