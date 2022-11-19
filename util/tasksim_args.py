import hashlib
from dataclasses import dataclass, fields, asdict
from argparse import ArgumentParser
from util.dataset_utils import DATASETS
import json
from models import PRETRAINED_MODELS

MODELS = ['resnet', 'densenet', 'vgg', 'vit', 'resnet_clip', 'vit_clip', 'efficientnet']
OPTIMS = ['adam', 'sgd']

@dataclass
class TaskSimArgs:
    results_dir: str = None
    batch_size: int = 32
    num_epochs: int = 1
    lr: float = 0.001
    freeze_features: bool = False
    head_type: str = 'linear'
    n_tasks: int = 5
    n_classes_per_task: int = -1
    dataset: str = "cifar-10"
    multihead: bool = False
    wandb: bool = False
    optim: str = "sgd"
    model: str = "resnet"
    pretrained: bool = False
    seed: int = 0
    task2vec: bool = False
    save_results: bool = False
    save_embeddings: bool = False
    save_feature_extractor: bool = False
    replay_size_per_class: int = 0
    init: bool = False
    momentum: float = 0.9
    sweep_overwrite: bool = False
    no_masking: bool = False
    patience: int = -1
    val_all: bool = False ## Validate on all tasks seen so far
    mixup: bool = False
    mixup_lambda: float = 0.0 ## should be between [0, 1]
    mixup_type: str = 'task_data'
    task2vec_epochs: int = 10
    task2vec_combined_head: bool = False
    task2vec_fisher_with_test: bool = False
    domain_inc: bool = False
    metrics: bool = False

    def __str__(self):
        return json.dumps(
            asdict(self), 
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
            separators=(',', ' : ')
        )

    def get_run_id(self):
        return hashlib.md5(str(self).encode('utf-8')).digest().hex()

    def validate_args(self):
        if self.mixup:
            assert self.n_tasks == 2, "mixup only configured for 2 task setting"
            assert self.mixup_lambda <= 1.0 and self.mixup_lambda >= 0, " arg mixup_lambda should be in [0, 1]"
            assert self.replay_size_per_class == 0
            assert self.mixup_type in ['task_data', 'noise']
        if self.multihead and (self.nmc or self.freeze_features):
            raise ValueError("Can't enable multihead and (freeze features/nmc)")
        if self.model not in MODELS and self.model not in PRETRAINED_MODELS:
            raise ValueError(f"{self.model} not supported!")
        if self.dataset not in DATASETS.keys():
            raise ValueError(f"{self.dataset} not supported!")
        if self.optim not in OPTIMS:
            raise ValueError(f"{self.optim} not supported!")
        if self.init:
            assert self.task2vec == False
        if self.domain_inc:
            assert self.dataset == 'cifar-100' or self.dataset == 'tiny', 'Only Cifar-100 is supported for domain_inc setting!'
            assert self.n_tasks == 5
            if self.dataset == 'cifar-100':
                assert self.n_classes_per_task == 20
        assert self.head_type in ['nmc', 'linear']
        assert self.replay_size_per_class >= -1
        if self.n_classes_per_task:
            assert self.n_classes_per_task > 0

def parse_args() -> TaskSimArgs:
    parser = ArgumentParser(description='Tasksim')
    for field in fields(TaskSimArgs):
        name = field.name
        default = field.default
        name = f'--{name.replace("_", "-")}'
        if field.type == bool:
            parser.add_argument(name, action='store_true')
        else:
            parser.add_argument(name, default=default, type=type(default))
    args = parser.parse_args()
    return TaskSimArgs(**vars(args))