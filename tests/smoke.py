import torch
from torch import nn
from micro_adam.micro_adam import MicroAdam

def test_microadam_step_runs():
    model = nn.Linear(10, 1)
    criterion = nn.MSELoss()
    optimizer = MicroAdam(model.parameters(), lr=1e-3)

    x = torch.randn(8, 10)
    y = torch.randn(8, 1)

    output = model(x)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()

    # just check it doesn't crash and parameters are updated
    for p in model.parameters():
        assert p.grad is not None
