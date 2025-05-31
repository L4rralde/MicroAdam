from sklearn import datasets
from torch.utils.data import Dataset, random_split



class IrisDatset(Dataset):
    def __init__(self, transforms=None):
        self.iris = datasets.load_iris()
        self.transform = transforms

    def __len__(self):
        return len(self.iris.data)

    def __getitem__(self, idx):
        return self.iris.data[idx], self.iris.target[idx]


DATASET = IrisDatset()

__train_size = int(0.9 * len(DATASET))
__test_size = len(DATASET) - __train_size

train_dataset, test_dataset = random_split(
    DATASET,
    [__train_size, __test_size]
)
