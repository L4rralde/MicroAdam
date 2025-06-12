import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.models.resnet import resnet18
from tqdm import tqdm

from utils.utils import DEVICE
from micro_adam.micro_adam import MicroAdam
from adam.adam import Adam


def main():

    torch.cuda.memory._record_memory_history()
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

    for i, data in tqdm(enumerate(trainloader)):
        x, y = data
        optimizer.zero_grad()
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        pred = model(x)
        loss = loss_fn(pred, y)
        loss.backward()

        optimizer.step()
        if i==2:
            break

    torch.cuda.memory._dump_snapshot("adam_profile.pkl")
    torch.cuda.memory._record_memory_history(enabled=None)


if __name__ == '__main__':
    main()
