import torch
from torch import nn
import torchvision
import torchvision.transforms as transforms

from utils.utils import DEVICE
from micro_adam.micro_adam import MicroAdam
from train.train import train


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
        epochs = 10,
        device = DEVICE
    )


if __name__ == '__main__':
    main()
