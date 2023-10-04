"""
Trains a PyTorch model according to the user's configuration in "config.json".

Currently only supports image classifiers.

[WARNING!] The script should be run from the /scripts/ directory
to make sure all file paths are correct. If you would still like to run
from a different workin directory, adjust the variable 'new_cwd' in
import statements to your /scripts/ file path.

The script uses a config.json file in the /scripts/ directory to obtain
all the necessary information. The config includes seed, hyperparameters,
device, data loading, model and save settings.

Explanation of settings in config.json:


TODO:
    - Add config.json functionality by parsing (separate script? make sure obj types correct)
    - Allow user to specify config locations from terminal with arg
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

# Change cd to scripts and import other modules
new_cwd = "/lustre/BIF/nobackup/to001/thesis_MBF/scripts"
print(f"[INFO] Changing working directory to {new_cwd}")
os.chdir(new_cwd)
import data_setup, engine, model_builder

# Import config setting variables from config_parser
from config_parser import *

print("[INFO] Loading config.json was succesful!")


def main():
    # Set seeds for reproducibility
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)

    # Create DataLoaders with help from data_setup.py
    # train_loader, test_loader = data_setup.get_loaders(
    #     dataset=DATASET,
    #     img_dir=IMG_DIR,
    #     label_dir=LABEL_DIR,
    #     train_frac=TRAIN_FRAC,
    #     augs=TRANSFORMS,
    #     batch_size=BATCH_SIZE,
    #     num_workers=NUM_WORKERS,
    #     pin_memory=PIN_MEMORY,
    # )
    train_loader, test_loader = data_setup.MNIST_digit_loaders(
        BATCH_SIZE, NUM_WORKERS, PIN_MEMORY
    )

    # Create model with help from model_builder.py
    model = MODEL_TYPE(
        n_classes=N_CLASSES,
        bb_name=BB_NAME,
        bb_weights=BB_WEIGHTS,
        bb_freeze=BB_FREEZE,
    )

    # Start training with help from engine.py
    # Load model if requested
    if LOAD_MODEL == True:
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

    # Setup tqdm loop for progress bar over epochs
    epoch_loop = tqdm(range(NUM_EPOCHS))

    # Training loop for a number of epochs
    for epoch in epoch_loop:
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
            if (epoch + 1) % CHECKPOINT_FREQ == 0:  # Current epoch is epoch + 1
                checkpoint = {
                    "state_dict": model.state_dict(),
                    "optimizer": OPTIMIZER.state_dict(),
                }
                utils.save_checkpoint(
                    state=checkpoint,
                    target_dir=SAVE_MODEL_DIR,
                    model_name=SAVE_MODEL_NAME,
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
    if NUM_EPOCHS % CHECKPOINT_FREQ != 0:  # Don't save when final epoch was checkpoint
        final_state = {
            "state_dict": model.state_dict(),
            "optimizer": OPTIMIZER.state_dict(),
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
        filename=f"results_{SAVE_MODEL_NAME}.tsv",
    )

    # Save a torchinfo summary of the network
    utils.save_network_summary(
        model=model,
        target_dir=SAVE_MODEL_DIR,
        filename=f"summary_{SAVE_MODEL_NAME}.txt",
    )

    # Save the config
    utils.save_config(
        target_dir=SAVE_MODEL_DIR,
        filename=f"config_{SAVE_MODEL_NAME}.json",
    )


if __name__ == "__main__":
    main()
