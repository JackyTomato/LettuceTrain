"""
Parses "_config_dict.json" to extract all settings as variables.

Extraction is done in such a way that variables are assigned
the settings as their correct object types, using eval().
E.g. the string "torch.optim.Adam" will be the PyTorch function torch.optim.Adam.

In "train.py" all variables that were assigned settings will be imported using
"from config_parser import *". Variables that were not assigned settings
are prefixed with "_" so they are not imported to train.py.

TODO:
    - Instead of .json make more readable .txt file to parse
"""
# Import statements
import os
import torch
import torch.nn as nn
import albumentations as A
from albumentations.pytorch import ToTensorV2
from json import load as json_load

# Import supporting modules
if "scripts" in os.getcwd():
    import data_setup, engine, model_builder, utils
else:
    # Change wd to scripts if cwd is not scripts
    new_cwd = "/lustre/BIF/nobackup/to001/thesis_MBF/scripts"
    print(f"[INFO] Changing working directory to {new_cwd}")
    os.chdir(new_cwd)
    import data_setup, engine, model_builder, utils


# .json parser for config.json
def parse_json(filepath):
    """Parses the a .json file as a dictionary containing all the values.

    Args:
        filepath (str): File path to the .json file.
    """
    # Parse config.json as dict with json's load function
    with open(filepath, "r") as json_file:
        json_dict = json_load(json_file)
    return json_dict


# Parse config.json as dict
_config_dict = parse_json("config.json")

# Assign settings to variables
# Seed
SEED = _config_dict["SEED"]

# Setup hyperparameters and other training specifics
LEARNING_RATE = _config_dict["LEARNING_RATE"]
NUM_EPOCHS = _config_dict["NUM_EPOCHS"]
OPTIMIZER = eval(_config_dict["OPTIMIZER"])
SCALER = eval(_config_dict["SCALER"])
LOSS_FN = eval(_config_dict["LOSS_FN"])
PERFORMANCE_FN = eval(_config_dict["PERFORMANCE_FN"])

# Setup device settings
DEVICE = _config_dict["DEVICE"]
NUM_WORKERS = _config_dict["NUM_WORKERS"]
PIN_MEMORY = eval(_config_dict["PIN_MEMORY"])

# Setup data loading settings
DATASET = eval(_config_dict["DATA_CLASS"])
IMG_DIR = _config_dict["IMG_DIR"]
LABEL_DIR = _config_dict["LABEL_DIR"]
TRAIN_FRAC = _config_dict["TRAIN_FRAC"]
TRAIN_TRANSFORMS = A.Compose([eval(_tf) for _tf in _config_dict["TRAIN_TRANSFORMS"]])
TEST_TRANSFORMS = A.Compose([eval(_tf) for _tf in _config_dict["TEST_TRANSFORMS"]])
BATCH_SIZE = _config_dict["BATCH_SIZE"]

# Setup model settings
MODEL_TYPE = eval(_config_dict["MODEL_TYPE"])
N_CLASSES = _config_dict["N_CLASSES"]
N_CHANNELS = _config_dict["N_CHANNELS"]
BB_NAME = _config_dict["BB_NAME"]
BB_WEIGHTS = _config_dict["BB_WEIGHTS"]
BB_FREEZE = eval(_config_dict["BB_FREEZE"])

# Setup checkpointing, save and load
CHECKPOINT_FREQ = _config_dict["CHECKPOINT_FREQ"]
SAVE_MODEL_DIR = _config_dict["SAVE_MODEL_DIR"]
SAVE_MODEL_NAME = _config_dict["SAVE_MODEL_NAME"]
LOAD_MODEL = eval(_config_dict["LOAD_MODEL"])
LOAD_MODEL_PATH = _config_dict["LOAD_MODEL_PATH"]
