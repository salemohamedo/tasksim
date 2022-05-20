import argparse
import time

from torchvision import models, transforms
from continuum import ClassIncremental
from continuum.datasets import CIFAR100, CIFAR10
from continuum.tasks import split_train_val
import torch
import numpy, random
from tqdm import tqdm

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
parser.add_argument('--num-epochs', type=int, default=1000, metavar='N',
                    help='number of epochs to train (default: 64)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--freeze-features', action='store_true',
                    help='Only train classifier head')
parser.add_argument('--increment', type=int, default=10, metavar='N')

args = parser.parse_args()
print(args)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )])


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
        # super().__init__(*args, **kwargs)

    def forward(self, x, *args, **kwargs):
        # return torch.nn.functional.linear(x, self.weight / torch.norm(self.weight, dim=1, keepdim=True), self.bias)
        return torch.nn.functional.linear(x, self.weight, self.bias)

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
    
    def restore_head_weights(self):
        if self.old_head_weights is not None:
            size = self.old_head_weights.size(0)
            print(size)
            print(self.old_head_weights.shape)
            print(self.encoder.fc.weight.data.shape)
            self.encoder.fc.weight.data[:size] = self.old_head_weights
            self.encoder.fc.bias.data[:size] = self.old_head_bias
    
    def forward(self, x, y):
        outs = self.encoder(x)
        classes_mask = torch.eye(self.head_size).cuda().float()
        label_unique = y.unique()
        ind_mask = classes_mask[label_unique].sum(0)
        full_mask = ind_mask.unsqueeze(0).repeat(outs.shape[0], 1)
        outs = torch.mul(outs, full_mask)
        return outs


def train(model, train_loader, optim: torch.optim.Optimizer, criterion: torch.nn.CrossEntropyLoss, prev_test_loaders):
    model.train()
    total_loss = 0
    for inputs, labels, task_ids in train_loader:
        # if len(prev_test_loaders) > 0:
        #     loss, acc = evaluate(model, prev_test_loaders[0], criterion)
        #     print(f"##Task: 0\tLoss: {loss:.4f}\tAcc: {acc*100:.2f}%\tLR: {None}")
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
        for inputs, labels, task_ids in test_loader:
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
        evaluate(model, test_loader, criterion)
        # lr_scheduler.step()

def prepare_scenarios(transform):
    train_dataset = CIFAR10('./data', train=True, download=True)
    train_scenario = ClassIncremental(
        train_dataset,
        increment=args.increment,
        initial_increment=0,
        transformations=[transform]
    )
    test_dataset = CIFAR10('./data', train=False, download=True)
    test_scenario = ClassIncremental(
        test_dataset,
        increment=args.increment,
        initial_increment=0,
        transformations=[transform]
    )
    return train_scenario, test_scenario

## Load data
train_scenario, test_scenario = prepare_scenarios(transform)

print(f"Number of classes: {train_scenario.nb_classes}.")
print(f"Number of tasks: {train_scenario.nb_tasks}.")

## Configure loss, optimizer, lr scheduler
model = PretrainedModel(args.freeze_features)
criterion = torch.nn.CrossEntropyLoss()

start_train_time = time.time()
prev_test_loaders = []
for task_id, (train_taskset, test_taskset) in enumerate(zip(train_scenario, test_scenario)):
    print(f"\n######\t Training on task {task_id}\t ######\n")
    ## Load data
    train_taskset, val_taskset = split_train_val(train_taskset, val_split=0.1)
    train_loader = torch.utils.data.DataLoader(
        train_taskset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    val_loader = torch.utils.data.DataLoader(
        val_taskset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_loader = torch.utils.data.DataLoader(
        test_taskset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    train_ys = set(train_taskset._y)
    test_ys = set(test_taskset._y)
    assert train_ys == test_ys
    # print(f"Labels for task {task_id}: {train_ys}")
    ## Update model, optimizer
    model.extend_head(train_taskset.nb_classes)
    optim_params = model.encoder.fc.parameters() if args.freeze_features else model.encoder.parameters()
    lr = args.lr
    # if task_id > 0:
    #     lr = 0.0001
    optim = torch.optim.SGD(optim_params, lr=lr,
                            momentum=0.9, weight_decay=5e-4)
    # Decay LR by a factor of 0.1 every 7 epochs
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=7, gamma=0.1)
    print(f"Model head size: {model.head_size}")
    # if task_id == 0:
    run_train_loop(model, train_loader, test_loader, optim, lr_scheduler, criterion, args.num_epochs, prev_test_loaders)
    # model.restore_head_weights()
    loss, acc = evaluate(model, test_loader, criterion)
    print(f"Task: {task_id}\tLoss: {loss:.4f}\tAcc: {acc*100:.2f}%\tLR: {None}")
    if len(prev_test_loaders) > 0:
        print(f"\n## Testing Previous Task Accuracies ##\n")
        for i, tl in enumerate(prev_test_loaders):
            loss, acc = evaluate(model, tl, criterion)
            print(f"Task: {i}\tLoss: {loss:.4f}\tAcc: {acc*100:.2f}%\tLR: {None}")
            # print(f"Labels for task {i}: {set(tl.dataset._y)}")
    prev_test_loaders.append(test_loader)


print(f"\n\nTrain time: {time.time()-start_train_time:.2f}s")
print(f"Total run time: {time.time()-very_start:.2f}s")
