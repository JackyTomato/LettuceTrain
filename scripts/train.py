"""
Trains a PyTorch image classification model using device-agnostic code.

[WARNING!] The script should be ran from the /scripts/ directory
to make sure all file paths are correct. If you would still like to run
from a different workin directory, adjust the variable 'new_cwd' in
import statements to your /scripts/ file path.

TODO:
    - Add config.json functionality by parsing (separate script? make sure obj types are correct)
    - Think about how to allow customizability transforms
    - Make torchvision summary extract input size from data
    - Also save loss and performance during training & testing
    - Also save model summary
    - Also save config
    - Create support for more than just classifiers (separate scripts?)
"""

# Import statements
import os
import torch
import numpy as np
import random
from tqdm import tqdm
try:
    import data_setup, engine, model_builder, utils
except:
    new_cwd = "/lustre/BIF/nobackup/to001/thesis_MBF/scripts"
    print(f"[INFO] Changing working directory to {new_cwd}")
    os.chdir(new_cwd)
    import data_setup, engine, model_builder, utils


# Seed
SEED = 

# Setup hyperparameters and other training specifics
LEARNING_RATE = 
NUM_EPOCHS = 
MODEL_TYPE = 
DATA_CLASS = 
DATALOADER_TRAIN = 
DATALOADER_TEST = 
OPTIMIZER = 
SCALER = 
LOSS_FN = 
PERFORMANCE_FN = 

# Setup device settings
DEVICE = 
NUM_WORKERS = 
PIN_MEMORY = 

# Setup data loading settings
IMG_DIR = 
LABEL_DIR = 
TRAIN_FRAC = 
TRANSFORMS = 
BATCH_SIZE = 

# Setup model settings
MODEL_TYPE = 
BB_NAME = 
BB_WEIGHTS = 
BB_FREEZE = 

# Setup checkpointing, save and load
CHECKPOINT_FREQ = 
SAVE_MODEL_DIR = 
SAVE_MODEL_NAME = 
LOAD_MODEL_PATH = 

def main():
    # Set seeds for reproducibility
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)

    # Create DataLoaders with help from data_setup.py
    train_loader, test_loader, class_names = data_setup.get_loaders(
        img_dir=IMG_DIR,
        label_dir=LABEL_DIR,
        train_frac=TRAIN_FRAC,
        augs=TRANSFORMS,
        batch_size=BATCH_SIZE,
    )

    # Create model with help from model_builder.py
    model = MODEL_TYPE(
        n_classes=len(class_names),
        bb_name=BB_NAME,
        bb_weights=BB_WEIGHTS,
        bb_freeze=BB_FREEZE,
    )

    # Start training with help from engine.py
    # Load model if requested
    if LOAD_MODEL_PATH:
        utils.load_checkpoint(checkpoint=LOAD_MODEL_PATH, model=model)

    # Prepare optimizer
    OPTIMIZER = OPTIMIZER(params=model.parameters(), lr=LEARNING_RATE)

    # Create empty results dictionary for loss and performance during training loop
    results = {
        "epoch": [],
        "train_loss": [],
        "train_perform": [],
        "test_loss": [],
        "test_perform": [],
    }

    # Training loop for a number of epochs with tqdm progress bars
    for epoch in tqdm(range(NUM_EPOCHS)):
        train_loss, train_perform = engine.train_step(
            model=model,
            dataloader=train_loader,
            loss_fn=LOSS_FN,
            performance_fn=PERFORMANCE_FN,
            optimizer=OPTIMIZER,
            scaler=SCALER,
            device=DEVICE,
        )
        test_loss, test_perform = engine.test_step(
            model=model,
            dataloader=test_loader,
            loss_fn=LOSS_FN,
            performance_fn=PERFORMANCE_FN,
            device=DEVICE,
        )

        # Checkpoint model at a given frequency if requested
        if CHECKPOINT_FREQ:
            if (epoch+1) % CHECKPOINT_FREQ == 0: # Current epoch is epoch+1
                checkpoint = {
                    "state_dict": model.state_dict(), 
                    "optimizer":OPTIMIZER.state_dict(),
                }
                utils.save_checkpoint(
                    state=checkpoint,
                    target_dir=SAVE_MODEL_DIR,
                    model_name=SAVE_MODEL_NAME,
                )

        # Print out epoch number, loss and performance for this epoch
        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_perform: {train_perform:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_perform: {test_perform:.4f}"
        )

        # Update results dictionary
        results["epoch"].append(epoch+1)
        results["train_loss"].append(train_loss)
        results["train_perform"].append(train_perform)
        results["test_loss"].append(test_loss)
        results["test_perform"].append(test_perform)

    # Save the model with help from utils.py
    if NUM_EPOCHS % CHECKPOINT_FREQ != 0: # Don't save when final epoch was checkpoint
        final_state = {
            "state_dict": model.state_dict(), 
            "optimizer":OPTIMIZER.state_dict(),
        }
        utils.save_checkpoint(
            state=final_state,
            target_dir=SAVE_MODEL_DIR,
            model_name=SAVE_MODEL_NAME,
        )

    # Save loss and performance during training
    utils.save_train_results(
        dict_results=results,
        target_dir=SAVE_MODEL_DIR,
        filename="results_"+SAVE_MODEL_NAME+".tsv",
    )

    # Save a torchinfo summary of the network
    utils.save_network_summary(
        model=model,
        target_dir=SAVE_MODEL_DIR,
        filename="summary_"+SAVE_MODEL_NAME+".txt",
    )

    # Save the config
    utils.save_config(
        target_dir=SAVE_MODEL_DIR,
        filename="config_"+SAVE_MODEL_NAME+".json")


if __name__ == "__main__":
    main()