#!/usr/bin/env python3
"""
Analyzes different trained models using loss & performance plots and inference.
"""

# Import statements
import os
import torch
import matplotlib.pyplot as plt

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
    model_filename = os.path.basename(model_filepath)

    # Parse config.json as dict
    config_filename = f"config_{model_filename.split(os.extsep)[0]}.json"
    config_dict = utils.parse_json(config_filename)

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

    # Load saved model state into freshly initialized model
    utils.load_checkpoint(model_filename, model)
    return model
