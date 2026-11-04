import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from torchvision import datasets, transforms


class JsonFemnistDataset(Dataset):
    def __init__(self, xs, ys):
        self.x = torch.tensor(np.asarray(xs), dtype=torch.float32).reshape(-1, 1, 28, 28)
        self.y = torch.tensor(np.asarray(ys), dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def cifar10_partition(root, num_clients, alpha, seed):
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    train = datasets.CIFAR10(root=root, train=True, download=True, transform=tfm)
    test = datasets.CIFAR10(root=root, train=False, download=True, transform=tfm)
    labels = np.asarray(train.targets)
    rng = np.random.default_rng(seed)
    client_indices = [[] for _ in range(num_clients)]
    for cls in np.unique(labels):
        idx = np.where(labels == cls)[0]
        rng.shuffle(idx)
        proportions = rng.dirichlet(np.repeat(alpha, num_clients))
        cuts = (np.cumsum(proportions) * len(idx)).astype(int)[:-1]
        for cid, split in enumerate(np.split(idx, cuts)):
            client_indices[cid].extend(split.tolist())
    clients = [Subset(train, sorted(v)) for v in client_indices]
    return clients, test


def _load_leaf_jsons(folder):
    folder = Path(folder)
    users = {}
    for p in sorted(folder.glob('*.json')):
        with open(p, 'r', encoding='utf-8') as f:
            obj = json.load(f)
        for u in obj['users']:
            users[u] = obj['user_data'][u]
    return users


def femnist_partition(root, num_clients):
    root = Path(root)
    train_users = _load_leaf_jsons(root / 'train')
    test_users = _load_leaf_jsons(root / 'test')
    common = sorted(set(train_users).intersection(test_users))[:num_clients]
    if len(common) < num_clients:
        raise RuntimeError(f'Need {num_clients} FEMNIST writers, found {len(common)} in {root}.')
    train_clients = [JsonFemnistDataset(train_users[u]['x'], train_users[u]['y']) for u in common]
    # concatenate selected writers' test samples for a global evaluation set
    xs, ys = [], []
    for u in common:
        xs.extend(test_users[u]['x']); ys.extend(test_users[u]['y'])
    test = JsonFemnistDataset(xs, ys)
    return train_clients, test
