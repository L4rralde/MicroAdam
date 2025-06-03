# MicroAdam


Check our implementation of `MicroAdam` in pure pytorch in [micro_adam.py](https://github.com/L4rralde/MicroAdam/blob/main/src/micro_adam/micro_adam.py). Currently only supports CPU instructions.

We also implemented `Adam` to do fair runtime and memory comparisons. Check [adam.py](https://github.com/L4rralde/MicroAdam/blob/main/src/adam/adam.py).

## How to install

1. Create a virtual environment, e.g.,

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install python modules

```
pip install -r requirements.txt
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