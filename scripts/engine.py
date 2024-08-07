#!/usr/bin/env python3
"""
Functionality for training and testing

TODO:
    - Add possibility of multiple training performance metrics
"""

# Import statements
import torch
from tqdm import tqdm


# Define train step function per epoch
def train_step(model, dataloader, loss_fn, performance_fn, optimizer, scaler, device):
    """Trains a PyTorch model for a single epoch.

    Turns a PyTorch model to training mode and then runs through all of the required training steps:
        1. forward pass, 2. loss calculation, 3. backward pass.

    The test step does not apply a sigmoid/softmax function, thus a loss function appropiate
    for the logits must be used.

    Uses automatic automatic mixed precision (AMP) training to utilize float16
    in order to speed up training while minimizing precision loss.
    AMP is performed using torch.cuda.amp.autocast() and a scaler.

    Args:
        model (nn.Module): A PyTorch model to be trained.
        dataloader (torch.utils.data.DataLoader): A DataLoader instance for the model to be trained on.
        loss_fn (nn.Module): A PyTorch loss function to minimize.
        performance_fn (function): A function that calculates a performance metric, e.g. class accuracy in utils.
            Alternatively a list of functions, to use multiple performance metrics.
        optimizer (torch.optim.Optimizer): A PyTorch optimizer to help minimize the loss function.
        scaler (torch.cuda.amp.GradScaler): A PyTorch gradient scaler to help minimize gradient underflow.
        device (torch.device): A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
        A tuple of training loss and training performance metrics.
        In the form (train_loss, train_performance). For example:

        (0.1112, 0.8743)

        Alternatively, train_performance can be a list when using multiple metrics, e.g.:
        (0.1112, [0.8743, 0.3245, 10])
    """
    # Put model in train mode
    model.train()

    # Setup tdqm loop for progress bar over batches
    batch_loop = tqdm(dataloader, desc="Batches")

    # Always treating performance_fn as list to allow for multiple metrics
    if not isinstance(performance_fn, list):
        performance_fn = [performance_fn]

    # Setup train loss and train performance values
    train_loss = 0
    train_perform = [0] * len(performance_fn)

    # Loop through data loader data batches
    for batch, (data, labels) in enumerate(batch_loop):
        # Add channel dimension or class dimension to label if not present yet
        if (len(labels.shape) == 3) or (len(labels.shape) == 1):
            labels = labels.unsqueeze(1)

        # Send data to target device
        data, labels = data.float().to(device), labels.float().to(device)

        # 1. Forward pass
        with torch.cuda.amp.autocast():
            pred_logits = model(data)

            # 2. Calculate  and accumulate loss
            loss = loss_fn(pred_logits, labels)
            train_loss += loss.item()

        # 3. Backward pass
        optimizer.zero_grad()  # Sets gradient to zero
        scaler.scale(loss).backward()  # Calculate gradient
        scaler.step(optimizer)  # Updates weights using gradient
        scaler.update()

        # Calculate and accumulate performance metrics across all batches
        for ind, fn in enumerate(performance_fn):
            train_perform[ind] += fn(pred_logits, labels)

        # Update tqdm loop
        batch_loop.set_postfix(train_loss=f"{loss.item():.4f}")

    # Adjust metrics to get average loss and performance per batch
    train_loss = train_loss / len(dataloader)
    train_perform = [perform / len(dataloader) for perform in train_perform]

    return train_loss, train_perform


# Define test step function per epoch
def test_step(model, dataloader, loss_fn, performance_fn, device):
    """Tests a PyTorch model for a single epoch.

    Turns a PyTorch model to eval mode and then performs a forward pass on a testing dataset.
    The test step does not apply a sigmoid/softmax function, thus a loss function appropiate
    for the logits must be used.

    Args:
        model (nn.Module): A PyTorch model to be tested.
        dataloader (torch.utils.data.DataLoader): A DataLoader instance for the model to be tested on.
        loss_fn (nn.Module): A PyTorch loss function to calculate loss on the test data.
        performance_fn (function): A function that calculates a performance metric, e.g. class accuracy in utils.
            Alternatively a list of functions, to use multiple performance metrics.
        device (torch.device): A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
        A tuple of testing loss and testing performance metrics.
        In the form (test_loss, test_accuracy). For example:

        (0.0223, 0.8985)

        Alternatively, train_performance can be a list when using multiple metrics, e.g.:
        (0.0223, [0.8985, 0.3385, 21])
    """
    # Put model in eval mode
    model.eval()

    # Always treating performance_fn as list to allow for multiple metrics
    if not isinstance(performance_fn, list):
        performance_fn = [performance_fn]

    # Setup test loss and test performance values
    test_loss = 0
    test_perform = [0] * len(performance_fn)

    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        for batch, (data, labels) in enumerate(dataloader):
            # Add channel dimension or class dimension to label if not present yet
            if (len(labels.shape) == 3) or (len(labels.shape) == 1):
                labels = labels.unsqueeze(1)

            # Send data to target device
            data, labels = data.float().to(device), labels.float().to(device)

            # 1. Forward pass
            test_pred_logits = model(data)

            # 2. Calculate and accumulate loss
            loss = loss_fn(test_pred_logits, labels)
            test_loss += loss.item()

            # Calculate and accumulate performance
            for ind, fn in enumerate(performance_fn):
                test_perform[ind] += fn(test_pred_logits, labels)

    # Adjust metrics to get average loss and performance per batch
    test_loss = test_loss / len(dataloader)
    test_perform = [perform / len(dataloader) for perform in test_perform]

    return test_loss, test_perform
