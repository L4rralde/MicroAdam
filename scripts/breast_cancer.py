import torch
from torch import nn
from torch.utils.data import DataLoader

from utils.utils import DEVICE
from micro_adam.micro_adam import MicroAdam
from dataset.breast_cancer import train_dataset, test_dataset


class Mlp(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(30, 8),
            nn.ReLU(),
            nn.Linear(8, 2),
        )

    def forward(self, x):
        x = x.float()
        return self.layer(x)


def train(
    model: nn.Module,
    train_dataloader: object,
    val_dataloader: object,
    loss_fn: object,
    optimizer: object,
    epochs=100
):
    train_losses = []
    val_losses = []
    for epoch in range(1, epochs + 1):
        train_loss = 0.0
        num_train_elements = 0
        model.train()
        for x, y in train_dataloader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)
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
        num_val_elements = 0
        model.eval()
        with torch.no_grad():
            for x, y in val_dataloader:
                x = x.to(DEVICE)
                y = y.to(DEVICE)
                pred = model(x)
                loss = loss_fn(pred, y)

                batch_size = x.size(0)
                val_loss += loss.data.item() * batch_size
                num_val_elements += batch_size
        val_loss /= num_val_elements
        val_losses.append(val_losses)
        print(f"Epoch: {epoch}. Training loss: {train_loss: .3e}. Validation loss: {val_loss: .3e}")

    return train_losses, val_losses


def main():
    model = Mlp().to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = MicroAdam(model.parameters(), lr=1e-3)

    trainloader = DataLoader(train_dataset, batch_size=16)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False)

    train(
        model = model,
        train_dataloader = trainloader,
        val_dataloader = testloader,
        loss_fn = loss_fn,
        optimizer = optimizer,
        epochs = 500
    )


if __name__ == '__main__':
    main()
