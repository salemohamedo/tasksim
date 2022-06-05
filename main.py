import argparse
import time

from continuum import ClassIncremental
from continuum.generators import ClassOrderGenerator
from continuum.tasks import split_train_val
import torch
import numpy as np, random

from similarity_metrics.task2vec import Task2Vec, get_model
from dataset_utils import DATASETS, load_dataset, get_transform
from models import PretrainedModel
from utils import get_run_id, save_results

import wandb

# wandb.init(project="tasksim", entity="omar-s")

very_start = time.time()

## REMOVE DETERMINISM
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
def seed_worker(worker_id):
    np.random.seed(0)
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
parser.add_argument('--nmc', action='store_true',
                    help='Measure NMC accuracy')
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

def train(model, train_loader, optim: torch.optim.Optimizer, criterion: torch.nn.CrossEntropyLoss, epoch):
    model.train()
    total_loss = 0
    for inputs, *labels in train_loader:
        labels = labels[0]
        inputs, labels = inputs.to(device), labels.to(device)
        if not model.nmc:
            optim.zero_grad()
        outs = model(inputs, labels)
        if model.nmc:
            model.encoder.fc.update_means(labels, epoch)
        loss = criterion(outs, labels)
        total_loss += loss
        if not model.nmc:
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


def run_train_loop(model, train_loader, test_loader, optim, lr_scheduler, criterion, num_epochs):
    for i in range(num_epochs):
        train(model, train_loader, optim, criterion, i)
        if not args.skip_eval:
            loss, acc = evaluate(model, test_loader, criterion)
            print(f"Loss: {loss:.4f}\tAcc: {acc*100:.2f}%")
        if not model.nmc:
            lr_scheduler.step()

def run_cl_sequence(model, train_scenario, test_scenario, nmc=False):
    ## Configure loss, optimizer, lr scheduler
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
        # optim = torch.optim.Adam(optim_params, lr=3e-4, weight_decay=5e-4)
        optim, lr_scheduler = None, None
        if not nmc:
            optim_params = model.encoder.fc.parameters() if args.freeze_features else model.encoder.parameters()
            optim = torch.optim.SGD(optim_params, lr=args.lr,
                                    momentum=0.9, weight_decay=5e-4)
            lr_scheduler = torch.optim.lr_scheduler.StepLR(
                optim, step_size=7, gamma=0.1)
        # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #     optim, T_max=args.num_epochs)

        num_epochs = 1 if nmc == True else args.num_epochs
        ## Train and evaluate test accuracy on current task
        run_train_loop(model, train_loader, val_loader, optim,
                       lr_scheduler, criterion, num_epochs)
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
        if not nmc:
            model.unfreeze_features()
            embeddings.append(Task2Vec(model.encoder, max_samples=1000,
                                        skip_layers=0).embed2(train_taskset))
            model.freeze_features()
    print(f"\n\nTrain time: {time.time()-start_train_time:.2f}s")
    return results, embeddings

def prepare_scenario(dataset, increment, transform, order=None):
    return ClassIncremental(
        dataset,
        increment=increment,
        transformations=transform,
        class_order=order
    )
transform = get_transform(args.dataset)
train_dataset, test_dataset = load_dataset(args.dataset)
train_scenario = prepare_scenario(train_dataset, args.increment, transform)
scenario_generator = ClassOrderGenerator(train_scenario)
seen_perms = set()

run_id = get_run_id()

print(f"Starting run: {run_id}")
    
for scenario_id in range(args.num_permutations):
    while tuple(train_scenario.class_order) in seen_perms:
        train_scenario = scenario_generator.sample(seed=scenario_id)
    seen_perms.add(tuple(train_scenario.class_order))
    test_scenario = prepare_scenario(test_dataset, args.increment, transform, train_scenario.class_order)
    model = PretrainedModel(device, args.freeze_features)
    model = model.to(device)
    nmc_results, linear_classifier_results, embeddings = None, None, None
    linear_classifier_results, embeddings = run_cl_sequence(model, train_scenario, test_scenario)
    if args.nmc:
        model.configure_nmc()
        nmc_results, _ = run_cl_sequence(model, train_scenario, test_scenario, nmc=True)
    if args.save_results:
        save_results(args, linear_classifier_results, nmc_results,
                     embeddings, run_id, scenario_id)


print(f"Total number of classes: {train_scenario.nb_classes}.")
print(f"Number of tasks: {train_scenario.nb_tasks}.")
print(f"Total run time: {time.time()-very_start:.2f}s")
