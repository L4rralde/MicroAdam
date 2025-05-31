from sklearn.datasets import load_breast_cancer
from torch.utils.data import Dataset, random_split



class BreastCancerDatset(Dataset):
    def __init__(self, transforms=None):
        self.data = load_breast_cancer()
        self.transform = transforms

    def __len__(self):
        return len(self.data.data)

    def __getitem__(self, idx):
        return self.data.data[idx], self.data.target[idx]


DATASET = BreastCancerDatset()

__train_size = int(0.9 * len(DATASET))
__test_size = len(DATASET) - __train_size

train_dataset, test_dataset = random_split(
    DATASET,
    [__train_size, __test_size]
)
