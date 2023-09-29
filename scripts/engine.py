"""
Functionality for training and testing

TODO:
    - Implement checkpointing in loop
    - Maybe add training plot
"""

# Import statements
import torch
import torch.nn as nn
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
        optimizer (torch.optim.Optimizer): A PyTorch optimizer to help minimize the loss function.
        scaler (torch.cuda.amp.GradScaler): A PyTorch gradient scaler to help minimize gradient underflow.
        device (torch.device): A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
        A tuple of training loss and training performance metrics.
        In the form (train_loss, train_performance). For example:

        (0.1112, 0.8743)
    """
    # Put model in train mode
    model.train()

    # Setup train loss and train performance values
    train_loss, train_perform = 0, 0

    # Setup tdqm loop for progress bar
    loop = tqdm(dataloader)

    # Loop through data loader data batches
    for batch, (data, labels) in enumerate(loop):
        # Send data to target device
        data, labels = data.to(device), labels.to(device)

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

        # Calculate and accumulate performance metric across all batches
        train_perform += performance_fn(pred_logits, labels)

        # Update tqdm loop
        loop.set_postfix(loss=loss.item())

    # Adjust metrics to get average loss and performance per batch
    train_loss = train_loss / len(dataloader)
    train_perform = train_perform / len(dataloader)
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
        device (torch.device): A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
        A tuple of testing loss and testing performance metrics.
        In the form (test_loss, test_accuracy). For example:

        (0.0223, 0.8985)
    """
    # Put model in eval mode
    model.eval()

    # Setup test loss and test performance values
    test_loss, test_perform = 0, 0

    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        for batch, (data, labels) in enumerate(dataloader):
            # Send data to target device
            data, labels = data.to(device), labels.to(device)

            # 1. Forward pass
            test_pred_logits = model(data)

            # 2. Calculate and accumulate loss
            loss = loss_fn(test_pred_logits, labels)
            test_loss += loss.item()

            # Calculate and accumulate performance
            test_perform += performance_fn(test_pred_logits, labels)

    # Adjust metrics to get average loss and performance per batch
    test_loss = test_loss / len(dataloader)
    test_perform = test_perform / len(dataloader)
    return test_loss, test_perform


# Define train function to loop over epochs
def train(
    model,
    train_dataloader,
    test_dataloader,
    optimizer,
    scaler,
    loss_fn,
    epochs,
    device,
):
    """Trains and tests a PyTorch model for a given number of epochs.

    Passes a PyTorch model through train_step() and test_step()
    functions for a number of epochs, training and testing the model
    in each iteration of the epoch loop.

    Calculates, prints and stores evaluation metrics throughout.

    Args:
        model (nn.Module): A PyTorch model to be trained and tested.
        train_dataloader (torch.utils.data.DataLoader): A DataLoader instance for the model to be trained on.
        test_dataloader (torch.utils.data.DataLoader): A DataLoader instance for the model to be tested on.
        optimizer (torch.optim.Optimizer): A PyTorch optimizer to help minimize the loss function.
        scaler (torch.cuda.amp.GradScaler): A PyTorch gradient scaler to help minimize gradient underflow.
        loss_fn (nn.Module): A PyTorch loss function to calculate loss on both datasets.
        epochs (int): An integer indicating how many epochs to train for.
        device (torch.device): A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
        A dictionary of training and testing loss as well as training and
        testing performance metrics. Each metric has a value in a list for
        each epoch.
        In the form: {train_loss: [...],
                        train_perform: [...],
                        test_loss: [...],
                        test_perform: [...]}
        For example if training for epochs=2:
                    {train_loss: [2.0616, 1.0537],
                        train_perform: [0.3945, 0.3945],
                        test_loss: [1.2641, 1.5706],
                        test_perform: [0.3400, 0.2973]}
    """
    # Create empty results dictionary
    results = {
        "train_loss": [],
        "train_perform": [],
        "test_loss": [],
        "test_perform": [],
    }

    # Loop through training and testing steps for a number of epochs with tqdm progress bar
    for epoch in tqdm(range(epochs)):
        train_loss, train_perform = train_step(
            model=model,
            dataloader=train_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
        )
        test_loss, test_perform = test_step(
            model=model,
            dataloader=test_dataloader,
            loss_fn=loss_fn,
            device=device,
        )

        # Print out what's happening
        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_perform: {train_perform:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_perform: {test_perform:.4f}"
        )

        # Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_perform"].append(train_perform)
        results["test_loss"].append(test_loss)
        results["test_perform"].append(test_perform)

    # Return the filled results at the end of the epochs
    return results
