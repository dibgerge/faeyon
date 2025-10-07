from torch.utils.data import Dataset


class FaeDataset(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return {
            "inputs": self.inputs[idx],
            "targets": self.targets[idx]
        }
