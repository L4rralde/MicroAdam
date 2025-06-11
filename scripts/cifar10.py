import torch
from torch import nn
import torchvision
import torchvision.transforms as transforms
from torchvision.models.resnet import resnet18

from utils.utils import DEVICE
from micro_adam.micro_adam import MicroAdam
from adam.adam import Adam
from train.train import train


def main():
    model = resnet18()
    model.fc.out_features = 10
    model = model.to(DEVICE)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = MicroAdam(model.parameters(), lr=1e-3)
    #optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    batch_size = 64
    trainset = torchvision.datasets.CIFAR10(root='./data/', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data/', train=False, download=True, transform=transform)
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
