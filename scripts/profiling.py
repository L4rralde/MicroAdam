import torch
import torch.nn as nn
import torch.profiler

from utils.utils import DEVICE
from micro_adam.micro_adam import MicroAdam
from adam.adam import Adam


def main():
    print("device:", DEVICE)
    model = nn.Linear(1024, 1024).to(DEVICE)
    optimizer = Adam(model.parameters())
    X = torch.randn(32, 1024).to(DEVICE)
    target = torch.randn(32, 1024).to(DEVICE)
    loss_fn = nn.MSELoss()

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        profile_memory=True,
        record_shapes=True,
    ) as prof:
        for step in range(20):
            optimizer.zero_grad()
            y_hat = model(X)
            loss = loss_fn(y_hat, target)
            loss.backward()
            optimizer.step()
            prof.step()
    print(prof.key_averages().table(sort_by="cuda_memory_usage"))

if __name__ == '__main__':
    main()
