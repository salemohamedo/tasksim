import argparse
import time
import re

from torchvision import models, transforms
from continuum import ClassIncremental
from continuum.generators import ClassOrderGenerator
from continuum.tasks import split_train_val
import torch
import numpy, random

from similarity_metrics.task2vec import Task2Vec, get_model
from dataset_utils import DATASETS, load_dataset

from pathlib import Path
import pickle
import pandas as pd
import numpy as np
import json

RESULTS_PATH = './results'

import wandb

# wandb.init(project="tasksim", entity="omar-s")

very_start = time.time()

## REMOVE DETERMINISM
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
def seed_worker(worker_id):
    # worker_seed = torch.initial_seed() % 2**32
    numpy.random.seed(0)
    random.seed(0)
g = torch.Generator()
g.manual_seed(0)
## REMOVE DETERMINISM

parser = argparse.ArgumentParser(description='Tasksim Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
                    help='input batch size for testing (default: 64)')
parser.add_argument('--num-epochs', type=int, default=1, metavar='N',
                    help='number of epochs to train (default: 1)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--freeze-features', action='store_true',
                    help='Only train classifier head')
parser.add_argument('--skip-eval', action='store_true',
                    help='Skip eval')
parser.add_argument('--save-results', action='store_true',
                    help='Save run results to ./results dir')
parser.add_argument('--increment', type=int, default=10, metavar='N')
parser.add_argument('--num-permutations', type=int, default=1, metavar='N')
parser.add_argument('--dataset', default="cifar-10", metavar='N', choices=DATASETS.keys())


args = parser.parse_args()
print(vars(args))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class WeightNorm_Classifier(torch.nn.Module):
    def __init__(self, in_dim, n_classes, bias=False):
        super().__init__()
        self.size_in, self.size_out = in_dim, n_classes
        self.weight = torch.nn.Parameter(torch.Tensor(n_classes, in_dim))
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(n_classes))
        else:
            self.bias = None

        # initialize weights
        torch.nn.init.kaiming_normal_(self.weight)  # weight init

    def forward(self, x, *args, **kwargs):
        return torch.nn.functional.linear(x, self.weight / torch.norm(self.weight, dim=1, keepdim=True), self.bias)
        # return torch.nn.functional.linear(x, self.weight, self.bias)


class NMC_Classifier(torch.nn.Module):
    """ Custom Linear layer but mimics a standard linear layer """


    def __init__(self, size_in, size_out, device='cpu', *args, **kwargs):
        super().__init__()
        self.device = device
        self.size_in, self.size_out = size_in, size_out
        self.data = torch.zeros(0, size_in)
        self.labels = torch.zeros(0)
        self.register_buffer('weight', torch.zeros(
            size_out, size_in))  # mean layer
        self.register_buffer('nb_inst', torch.zeros(size_out))
        self.register_buffer('_initiated', torch.Tensor([0.]))

    @property
    def initiated(self):
        if self._initiated.item() == 0.:
            return False
        else:
            return True

    @initiated.setter
    def initiated(self, value: bool):
        if value:
            self._initiated = torch.Tensor([1.])
        else:
            self._initiated = torch.Tensor([0.])

    def __call__(self, x, y=None, epoch=None, *args, **kwargs):
        self.to('cpu')
        x = x.to('cpu')
        o = self.forward(x)
        if self.training and y is not None and epoch is not None:
            assert y is not None
            assert epoch is not None
            self.accumulate(x, y, epoch)
        return o

    def forward(self, x):
        data = x.detach().cpu()  # no backprop possible

        assert not torch.isnan(data).any()

        if self.initiated:
            # torch.cdist(c * b, d * b) -> c*d
            out = torch.cdist(data, self.weight)
            # convert smaller is better into bigger in better
            out = out * -1
        else:
            # if mean are not initiate we return random predition
            out = torch.randn((data.shape[0], self.size_out)).to(self.device)
        return out.to(self.device)

    def update(self, epoch=0):
        pass

    @torch.no_grad()
    def accumulate(self, x, y, epoch=0):
        if epoch == 0:
            self.data = x.view(-1, self.size_in).cpu()
            self.labels = y
            for i in range(self.size_out):
                indexes = torch.where(self.labels == i)[0]
                self.weight[i] = (
                    self.weight[i] * (1.0 * self.nb_inst[i]) + self.data[indexes].sum(0))
                self.nb_inst[i] += len(indexes)
                if self.nb_inst[i] != 0:
                    self.weight[i] = self.weight[i] / (1.0 * self.nb_inst[i])

            self.data = torch.zeros(0, self.size_in)
            self.labels = torch.zeros(0)
            self.initiated = True

        assert not torch.isnan(self.weight).any()

    def expand(self, size_out):
        self.size_out = size_out
        weight = torch.zeros(self.size_out, self.size_in)
        weight[:self.weight.shape[0]] = self.weight
        self.register_buffer('weight', weight)  # mean layer
        nb_inst = torch.zeros(size_out)
        nb_inst[:self.nb_inst.shape[0]] = self.nb_inst
        self.register_buffer('nb_inst', nb_inst)
        # pass

## Configure model
class PretrainedModel(torch.nn.Module):
    def __init__(self, freeze_features=False):
        super().__init__()
        self.encoder = models.resnet34(pretrained=True)
        self.head_size = 0
        self.fc_in_features = self.encoder.fc.in_features
        self.old_head_weights, self.old_head_bias = None, None
        if freeze_features:
            for param in self.encoder.parameters():
                param.requires_grad = False
        
    def extend_head(self, n):
        new_head_size = self.head_size + n
        # new_head = torch.nn.Linear(self.fc_in_features, new_head_size, bias=False)
        new_head = WeightNorm_Classifier(self.fc_in_features, new_head_size, bias=True)
        if self.head_size != 0: ## Save old class weights
            self.old_head_weights = self.encoder.fc.weight.data.clone().to(device)
            self.old_head_bias = self.encoder.fc.bias.data.clone().to(device)            
            new_head.weight.data[:self.head_size] = self.encoder.fc.weight.data.clone().to(device)
            new_head.bias.data[:self.head_size] = self.encoder.fc.bias.data.clone(
            ).to(device)
        self.encoder.fc = new_head
        self.head_size = new_head_size
        self.encoder.to(device)
    
    def unfreeze_features(self):
        for name, param in self.encoder.named_parameters():
            if re.search("fc", name) is None:
                param.requires_grad = True
    
    def freeze_features(self):
        for name, param in self.encoder.named_parameters():
            if re.search("fc", name) is None:
                param.requires_grad = False
    
    def forward(self, x, y):
        outs = self.encoder(x)
        classes_mask = torch.eye(self.head_size).cuda().float()
        label_unique = y.unique()
        ind_mask = classes_mask[label_unique].sum(0)
        full_mask = ind_mask.unsqueeze(0).repeat(outs.shape[0], 1)
        outs = torch.mul(outs, full_mask)
        return outs

def get_run_id():
    run_id = 0
    results_dir = Path(RESULTS_PATH)
    if results_dir.exists():
        id_list = [int(str(x).split("_")[-1]) for x in results_dir.iterdir()]
        run_id = 0 if not id_list else max(id_list) + 1
    return run_id


def save_results(results, embeddings, run_id, scenario_id):
    results_dir = Path(RESULTS_PATH)
    if not results_dir.exists():
        results_dir.mkdir()
    run_dir = results_dir / f'run_{str(run_id).zfill(3)}'
    if not run_dir.exists():
        run_dir.mkdir()
        with open(run_dir / 'config.txt', 'w') as config:
            json.dump(vars(args), config)
    df = pd.DataFrame(results)
    df.to_csv(run_dir / f'case_{scenario_id}.acc', float_format='%.3f')
    with open(run_dir / f'case_{scenario_id}.emb', 'wb') as f:
        pickle.dump(embeddings, f, protocol=pickle.HIGHEST_PROTOCOL)

def train(model, train_loader, optim: torch.optim.Optimizer, criterion: torch.nn.CrossEntropyLoss, prev_test_loaders):
    model.train()
    total_loss = 0
    for inputs, *labels in train_loader:
        labels = labels[0]
        inputs, labels = inputs.to(device), labels.to(device)
        optim.zero_grad()
        outs = model(inputs, labels)
        loss = criterion(outs, labels)
        total_loss += loss
        loss.backward()
        optim.step()
    return


def evaluate(model, test_loader, criterion):
    model.eval()
    total_loss = 0
    accuracy = 0
    with torch.no_grad():
        for inputs, *labels in test_loader:
            labels = labels[0]
            inputs, labels = inputs.to(device), labels.to(device)
            outs = model(inputs, labels)
            loss = criterion(outs, labels)
            accuracy += torch.sum(outs.max(1)
                                  [1] == labels).float() / len(labels)
            total_loss += loss
    return total_loss/len(test_loader), accuracy/len(test_loader)


def run_train_loop(model, train_loader, test_loader, optim, lr_scheduler, criterion, num_epochs, prev_test_loaders):
    for i in range(num_epochs):
        train(model, train_loader, optim, criterion, prev_test_loaders)
        if not args.skip_eval:
            evaluate(model, test_loader, criterion)
        # lr_scheduler.step()

def run_cl_sequence(train_scenario, test_scenario):
    ## Configure loss, optimizer, lr scheduler
    model = PretrainedModel(args.freeze_features)
    criterion = torch.nn.CrossEntropyLoss()
    start_train_time = time.time()
    prev_test_loaders = []
    embeddings = []
    results = np.zeros((train_scenario.nb_tasks, train_scenario.nb_tasks))
    for task_id, (train_taskset, test_taskset) in enumerate(zip(train_scenario, test_scenario)):
        print(f"\n######\t Training on task {task_id}\t ######\n")

        ## Load data
        train_taskset, val_taskset = split_train_val(train_taskset, val_split=0.1)
        train_loader = torch.utils.data.DataLoader(
            train_taskset, batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=True, pin_memory=True)
        val_loader = torch.utils.data.DataLoader(
            val_taskset, batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=True, pin_memory=True)
        test_loader = torch.utils.data.DataLoader(
            test_taskset, batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=True, pin_memory=True)
        train_ys = set(train_taskset._y)
        test_ys = set(test_taskset._y)
        assert train_ys == test_ys

        ## Update model, optimizer
        model.extend_head(train_taskset.nb_classes)
        optim_params = model.encoder.fc.parameters() if args.freeze_features else model.encoder.parameters()
        # optim = torch.optim.Adam(optim_params, lr=3e-4, weight_decay=5e-4)
        optim = torch.optim.SGD(optim_params, lr=args.lr,
                                momentum=0.9, weight_decay=5e-4)
        # lr_scheduler = torch.optim.lr_scheduler.StepLR(
        #     optim, step_size=7, gamma=0.1)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim, T_max=args.num_epochs)


        ## Train and evaluate test accuracy on current task
        run_train_loop(model, train_loader, test_loader, optim, lr_scheduler, criterion, args.num_epochs, prev_test_loaders)
        loss, acc = evaluate(model, test_loader, criterion)
        results[task_id][task_id] = acc
        print(f"Task: {task_id}\tLoss: {loss:.4f}\tAcc: {acc*100:.2f}%\tLR: {None}")

        ## Eval test accuracy on previous tasks
        if task_id == train_scenario.nb_tasks - 1:
            print(f"\n## Testing Previous Task Accuracies ##\n")
            for i, tl in enumerate(prev_test_loaders):
                loss, acc = evaluate(model, tl, criterion)
                print(f"Task: {i}\tLoss: {loss:.4f}\tAcc: {acc*100:.2f}%\tLR: {None}")
                results[task_id][i] = acc
        prev_test_loaders.append(test_loader)

        # Compute task2vec embedding
        # probe_network = get_model('resnet34', pretrained=True, num_classes=int(
        #     train_taskset.nb_classes+1)).cuda()
        # train_taskset._y -= train_taskset._y.min()
        # embeddings.append(Task2Vec(probe_network, max_samples=1000,
        #                   skip_layers=6).embed(train_taskset))
        model.unfreeze_features()
        embeddings.append(Task2Vec(model.encoder, max_samples=1000,
                                    skip_layers=0).embed2(train_taskset))
        model.freeze_features()
    print(f"\n\nTrain time: {time.time()-start_train_time:.2f}s")
    return results, embeddings

if args.dataset == 'mnist':
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
    if args.dataset == 'cub200':
        transform.insert(0, transforms.Resize([224, 224]))

def prepare_scenario(dataset, increment, transform, order=None):
    
    return ClassIncremental(
        dataset,
        increment=increment,
        transformations=transform,
        class_order=order
    )

train_dataset, test_dataset = load_dataset(args.dataset)
train_scenario = prepare_scenario(train_dataset, args.increment, transform)
scenario_generator = ClassOrderGenerator(train_scenario)
seen_perms = set()

run_id = get_run_id()

print(f"Starting run: {run_id}")
    
for scenario_id in range(args.num_permutations):
    while tuple(train_scenario.class_order) in seen_perms:
        train_scenario = scenario_generator.sample(seed=scenario_id)
    test_scenario = prepare_scenario(test_dataset, args.increment, transform, train_scenario.class_order)
    results, embeddings = run_cl_sequence(train_scenario, test_scenario)
    seen_perms.add(tuple(train_scenario.class_order))

    if args.save_results:
        save_results(results, embeddings, run_id, scenario_id)

print(f"Total number of classes: {train_scenario.nb_classes}.")
print(f"Number of tasks: {train_scenario.nb_tasks}.")
print(f"Total run time: {time.time()-very_start:.2f}s")
