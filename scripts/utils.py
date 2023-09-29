"""
Contains various utility functions for PyTorch model training and saving.
"""
import torch
import torch.nn as nn
from pathlib import Path


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


def load_checkpoint(checkpoint, model):
    """Loads a saved PyTorch model state.

    Args:
      checkpoint (str): File of saved model state including path if necessary.
      model (nn.Module): PyTorch model on which the state will be applied.
    """
    print(f"[INFO] Opening model state: {checkpoint}")
    model.load_state_dict(checkpoint["state_dict"])


def class_accuracy(pred_logits, labels):
    """Calculate classification accuracy.

    Accuracy is determined as the fraction of predictions that match
    the ground-truth labels.

    Args:
      pred_logits (torch.Tensor): Tensor of prediction logits.
      labels (torch.Tensor): Tensor of ground truth labels.
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
