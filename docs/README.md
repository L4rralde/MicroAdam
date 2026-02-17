# MicroAdam


Check our implementation of [MicroAdam: Accurate Adaptive Optimization with Low Space Overhead and Provable Convergence](https://arxiv.org/pdf/2405.15593) in pure pytorch in [micro_adam.py](https://github.com/L4rralde/MicroAdam/blob/main/src/micro_adam/micro_adam.py).

We also implemented `Adam` to do fair runtime and memory comparisons. Check [adam.py](https://github.com/L4rralde/MicroAdam/blob/main/src/adam/adam.py).

## Technical report

Read our technical report in spanish at the following [link](microadam_report.pdf).


## Latest release notes:

- Tested with CPU, CUDA and MPS.


## How to install

1. Create a virtual environment, e.g.,

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install python modules

```bash
pip install -r requirements.txt
```

3. Setup `PYTHONPATH`

```bash
source .bashrc
```


## Runnin examples

**Mnist**

```bash
python scripts/mnist.py
```

**Iris**

```bash
python scripts/iris.py
```

**Breast cancer**

```bash
python scripts/breast_cancer.py
```

## Code at a glance

Let's take a look into [scripts/mnist.py](https://github.com/L4rralde/MicroAdam/blob/main/scripts/mnist.py)


1. Importing packages.

```python
import torch
from torch import nn
import torchvision
import torchvision.transforms as transforms

from utils.utils import DEVICE
from micro_adam.micro_adam import MicroAdam #Our implementation of the optimizer.
```

2. Neural Network to Train: MLP.

```python
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
```

3. Function to train an MLP for data classification

```python
def train(
    model: nn.Module,
    train_dataloader: object,
    val_dataloader: object,
    loss_fn: object,
    optimizer: object,
    epochs=100
):
    ...
```

4. Instantiate MLP, loss function and our optimizer (main code starts)
```python
model = Mlp().to(DEVICE)
loss_fn = nn.CrossEntropyLoss()
optimizer = MicroAdam(model.parameters(), lr=1e-3)
```

5. Load data sets.
```python
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

batch_size = 64
trainset = torchvision.datasets.MNIST(root='./data/', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

testset = torchvision.datasets.MNIST(root='./data/', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
```

6. (Finally) train

```python
train_losses, val_losses = train(
    model = model,
    train_dataloader = trainloader,
    val_dataloader = testloader,
    loss_fn = loss_fn,
    optimizer = optimizer,
    epochs = 10
)
```

## Future Work

- [ ] Check cuda memory use
- [ ] Train on larger problems.
- [ ] Normal float
