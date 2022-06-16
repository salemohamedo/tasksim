from continuum.datasets import CIFAR100, CIFAR10, MNIST, CUB200, FGVCAircraft, Car196
from continuum.datasets import ImageFolderDataset
from torchvision import transforms
from utils.dataset_classes import CIFAR100_taxonomy, Car196_taxonomy, CIFAR10_taxonomy, CUB200_taxonomy
import clip
# from models import CLIP_MODEL_NAME_DICT
CLIP_MODEL_NAME_DICT = {
    'resnet_clip': 'RN50',
    'vit_clip': 'ViT-B/16'
}
DATASETS = {
    "cifar-10": (CIFAR10, 10),
    "cifar-100": (CIFAR100, 100),
    "mnist": (MNIST, 10),
    "cub200": (CUB200, 200),
    "car196": (Car196, 196),
    "tiny": (None, 200)
}

DATASETS_TAX = {
    "cifar-10": CIFAR10_taxonomy,
    "cifar-100": CIFAR100_taxonomy,
    "cub200": CUB200_taxonomy,
    "car196": Car196_taxonomy
}

def get_dataset_class_names(dataset, class_ids):
    return [DATASETS_TAX[dataset][i] for i in class_ids]

def load_dataset(dataset, domain_inc):
    if dataset not in DATASETS:
        raise ValueError(f"Dataset: {dataset} not valid")

    if dataset == 'tiny':
        train_dataset = ImageFolderDataset('data/tiny-32/train')
        test_dataset = ImageFolderDataset('data/tiny-32/test', train=False)
    else:
        if domain_inc:
            labels_type = "category"
            task_labels = "lifelong"
        else:
            labels_type = "class"
            task_labels = None
        train_dataset = DATASETS[dataset][0]('./data', train=True, download=True, labels_type=labels_type, task_labels=task_labels)
        test_dataset = DATASETS[dataset][0](
            './data', train=False, download=True, labels_type=labels_type, task_labels=task_labels)
    train_dataset.num_classes = DATASETS[dataset][1]
    test_dataset.num_classes = DATASETS[dataset][1]
    return train_dataset, test_dataset

def get_transform(dataset, model):
    if "clip" in model:
        clip_model_name = CLIP_MODEL_NAME_DICT[model]
        _, transform = clip.load(clip_model_name)
        return [transform]
    elif dataset == 'mnist':
        transform = [
            transforms.Resize([32, 32]),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1))
        ]
    else:
        size = [100, 100]
        transform = [
            transforms.Resize([32, 32]),
            # transforms.RandomCrop(size),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize(
            #     mean=[0.485, 0.456, 0.406],
            #     std=[0.229, 0.224, 0.225])
        ]
        if dataset == 'cub200':
            transform.insert(0, transforms.Resize([224, 224]))
        elif dataset == 'car196':
            transform.insert(0, transforms.Resize([100, 100]))
    return transform
