"""
Contains various utility functions such as checkpointing and performance metrics.

TODO:
    - Test save results, save summary and save config
"""
# Import functions
import torch
import torch.nn as nn
from torchinfo import summary
from pathlib import Path
from shutil import copyfile


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
    model_save_path = target_dir_path / model_name

    # Save the model state_dict()
    print(f"[INFO] Saving model states to: {model_save_path}")
    torch.save(obj=state, f=model_save_path)
    print(f"[INFO] Saving model to {model_save_path} was succesful!")


def load_checkpoint(checkpoint, model):
    """Loads a saved PyTorch model state.

    Args:
        checkpoint (str): File of saved model state including path if necessary.
        model (nn.Module): PyTorch model on which the state will be applied.
    """
    print(f"[INFO] Opening model state: {checkpoint}")
    model.load_state_dict(torch.load(checkpoint)["state_dict"])
    print(f"[INFO] Loading model state {checkpoint} was succesful!")


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


# Save training results
def save_train_results(dict_results, target_dir, filename):
    """Writes training loop results to a tab-delimited text file.

    Input dictionary of results should be in format as constructed in train.py.
    Thus, with "epoch", "train_loss", "train_perform", "test_loss" and "test_perform"
    as keys.

    Args:
        dict_results (dict): Dictionary of training loop results.
        target_dir (string): Target directory in which to write the file.
        filename (string): File name of the text file to be written.
    """
    filepath = target_dir + "/" + filename

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


def save_network_summary(model, target_dir, filename):
    """Writes a torchinfo summary and raw print summary of a model to a text file

    Args:
        model (nn.Module): PyTorch model to be summarized.
        target_dir (string): Target directory in which to write the file.
        filename (string): File name of the text file to be written.
    """
    filepath = target_dir + "/" + filename

    # Get torchinfo summary of model
    model_stats = summary(
        model=model,
        input_size=(1, 3, 512, 512),
        col_names=["kernel_size", "input_size", "output_size", "num_params"],
    )
    str_model_stats = str(model_stats)

    # Get raw model as string (show all layers and parameters)
    str_raw_model = str(model)

    # Write model summaries to file
    with open(filepath, "w") as f:
        f.write("TORCHINFO SUMMARY")
        f.write("\n")
        f.write(str_model_stats)
        f.write("\n")
        f.write("RAW SUMMARY")
    print(f"[INFO] Saved network summaries to {filepath}")


def save_config(target_dir, filename):
    """Saves config.json used for the model to a new text file.

    Args:
        target_dir (string): Target directory in which to write the file.
        filename (string): File name of the text file to be written.
    """
    source = "config.json"
    filepath = target_dir + "/" + filename
    copyfile(source, filepath)
    print(f"[INFO] Saved config.json to {filepath}")
