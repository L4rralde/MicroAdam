from sklearn.datasets import load_breast_cancer
import numpy as np
from torch.utils.data import Dataset, random_split


class BreastCancerDatset(Dataset):
    def __init__(self, transforms=None):
        dataset = load_breast_cancer()
        X = dataset.data.astype(np.float32)
        self.X = (X - X.mean(axis=0))/X.std(axis=0)
        self.y = dataset.target
        self.transform = transforms

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


DATASET = BreastCancerDatset()

__train_size = int(0.9 * len(DATASET))
__test_size = len(DATASET) - __train_size

train_dataset, test_dataset = random_split(
    DATASET,
    [__train_size, __test_size]
)
