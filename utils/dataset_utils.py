from continuum.datasets import CIFAR100, CIFAR10, MNIST, CUB200, FGVCAircraft
from torchvision import transforms

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

def get_transform(dataset):
    if dataset == 'mnist':
        transform = [
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1))
        ]
    else:
        transform = [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
        ]
        if dataset == 'cub200':
            transform.insert(0, transforms.Resize([224, 224]))
    return transform
