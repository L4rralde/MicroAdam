import torch
from torch import nn
import torchvision
import torchvision.transforms as transforms

from utils.utils import DEVICE
from micro_adam.micro_adam import MicroAdam


class Mlp(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.layers = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 10)
        )
    
    def forward(self, x):
        x = self.flatten(x)
        return self.layers(x)


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
        val_losses.append(val_loss)
        print(f"Epoch: {epoch}. Training loss: {train_loss: .3e}. Validation loss: {val_loss: .3e}")

    return train_losses, val_losses


def main():
    model = Mlp().to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = MicroAdam(model.parameters(), lr=1e-3)
    

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    batch_size = 64
    trainset = torchvision.datasets.MNIST(root='./data/', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.MNIST(root='./data/', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    train(
        model = model,
        train_dataloader = trainloader,
        val_dataloader = testloader,
        loss_fn = loss_fn,
        optimizer = optimizer,
        epochs = 10
    )


if __name__ == '__main__':
    print(DEVICE)
    main()
