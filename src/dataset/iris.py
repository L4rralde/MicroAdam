from sklearn import datasets
import numpy as np
from torch.utils.data import Dataset, random_split


class IrisDataset(Dataset):
    def __init__(self, transforms=None):
        iris_ds = datasets.load_iris()
        self.X = iris_ds.data.astype(np.float32)
        self.y = iris_ds.target
        self.transform = transforms

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


DATASET = IrisDataset()

__train_size = int(0.9 * len(DATASET))
__test_size = len(DATASET) - __train_size

train_dataset, test_dataset = random_split(
    DATASET,
    [__train_size, __test_size]
)
