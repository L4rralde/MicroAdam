import torch
from torch import nn
from tqdm import tqdm


def train(
    model: nn.Module,
    train_dataloader: object,
    val_dataloader: object,
    loss_fn: object,
    optimizer: object,
    epochs: int=100,
    device: str="cpu",
    verbose: bool=True,
    save: bool=False
) -> tuple:
    print(f"Training on {device}")
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    for epoch in tqdm(range(1, epochs + 1)):
        train_loss = 0.0
        num_train_elements = 0
        model.train()
        for x, y in train_dataloader:
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            batch_size = x.size(0)
            train_loss += loss.data.item() * batch_size
            num_train_elements += batch_size
        
        train_loss /= num_train_elements
        train_losses.append(train_loss)

        val_loss = 0.0
        correct = 0
        num_val_elements = 0
        model.eval()
        with torch.no_grad():
            for x, y in val_dataloader:
                x = x.to(device)
                y = y.to(device)
                pred = model(x)
                loss = loss_fn(pred, y)
                correct += (torch.argmax(pred, axis=1) == y).sum()
                #print(y, torch.argmax(pred, axis=1))
                batch_size = x.size(0)
                val_loss += loss.data.item() * batch_size
                num_val_elements += batch_size
        val_loss /= num_val_elements
        val_losses.append(val_loss)
        accuracy = 100*correct/num_val_elements
        if verbose:
            print(f"Epoch: {epoch}. Training loss: {train_loss: .3e}. Validation loss: {val_loss: .3e}. Validation Accuracy: {accuracy: .2f}")
        if save and val_loss < best_val_loss:
            best_val_loss = val_loss
            model_name = f"{model.__class__.__name__}_{val_loss: .2f}.th"
            print("Saving model at:", model_name)
            torch.save(model.state_dict(), model_name)

    return train_losses, val_losses