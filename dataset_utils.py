from continuum.datasets import CIFAR100, CIFAR10, MNIST, CUB200, FGVCAircraft

DATASETS = {
    "cifar-10": CIFAR10,
    "cifar-100": CIFAR100,
    "mnist": MNIST,
    "cub200": CUB200,
    "fgvc-aircraft": FGVCAircraft
}

def load_dataset(dataset):
    if dataset not in DATASETS:
        raise ValueError(f"Dataset: {dataset} not valid")

    train_dataset = DATASETS[dataset]('./data', train=True, download=True)
    test_dataset = DATASETS[dataset]('./data', train=False, download=True)
    return train_dataset, test_dataset
