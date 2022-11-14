from continuum.datasets import CIFAR100, CIFAR10, MNIST, CUB200, FGVCAircraft, Car196
from continuum.datasets import ImageFolderDataset

from continuum.datasets import _ContinuumDataset, CIFAR10, TinyImageNet200, ImageFolderDataset
from continuum.scenarios import TransformationIncremental
from continuum.tasks import TaskSet, TaskType
import numpy as np
from typing import Tuple, List, Union

from torchvision import transforms
from util.dataset_classes import CIFAR100_taxonomy, Car196_taxonomy, CIFAR10_taxonomy, CUB200_taxonomy
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


class CustomDomainIncDataset(_ContinuumDataset):
    """Continuum dataset for datasets with tree-like structure.

    :param train_folder: The folder of the train data.
    :param test_folder: The folder of the test data.
    :param download: Dummy parameter.
    """

    def __init__(
            self,
            data,
            data_path,
            train: bool = True,
            download: bool = True,
            data_type: TaskType = TaskType.IMAGE_PATH
    ):
        self.data_path = data_path
        self.data = data
        self._data_type = data_type
        super().__init__(data_path=data_path, train=train, download=download)

    @property
    def data_type(self) -> TaskType:
        return self._data_type

    def get_data(self) -> Tuple[np.ndarray, np.ndarray, Union[None, np.ndarray]]:
        # self.dataset = torchdata.ImageFolder(self.data_path)
        # x, y, t = self._format(self.dataset.imgs)
        # self.list_classes = np.unique(y)
        return self.data[0], self.data[1], self.data[2]

    @staticmethod
    def _format(raw_data: List[Tuple[str, int]]) -> Tuple[np.ndarray, np.ndarray, None]:
        x = np.empty(len(raw_data), dtype="S255")
        y = np.empty(len(raw_data), dtype=np.int64)

        for i, (path, target) in enumerate(raw_data):
            x[i] = path
            y[i] = target

        return x, y, None

def get_dataset_class_names(dataset, class_ids):
    return [DATASETS_TAX[dataset][i] for i in class_ids]


def load_dataset(dataset, domain_inc,  n_classes_per_task=None):
    if dataset not in DATASETS:
        raise ValueError(f"Dataset: {dataset} not valid")

    if dataset == 'tiny':
        train_dataset = ImageFolderDataset('data/tiny-32/train')
        test_dataset = ImageFolderDataset('data/tiny-32/test', train=False)
        if domain_inc:
            train_class_ids = np.array(train_dataset.get_data()[
                                       1] < n_classes_per_task)
            train_data = [x[train_class_ids]
                          for x in train_dataset.get_data() if x is not None]
            train_data.append(None)
            train_dataset = CustomDomainIncDataset(train_data, 'data/tiny-32/train', train=True)

            test_class_ids = np.array(
                test_dataset.get_data()[1] < n_classes_per_task)
            test_data = [x[test_class_ids]
                         for x in test_dataset.get_data() if x is not None]
            test_data.append(None)
            test_dataset = CustomDomainIncDataset(
                test_data, 'data/tiny-32/test', train=False)
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
