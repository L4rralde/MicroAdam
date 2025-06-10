import torch
from torch import nn
from torch.utils.data import DataLoader

from utils.utils import DEVICE
from micro_adam.micro_adam import MicroAdam
from dataset.breast_cancer import train_dataset, test_dataset
from train.train import train


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
        epochs = 1000,
        device = DEVICE
    )


if __name__ == '__main__':
    main()
