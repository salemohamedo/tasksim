import argparse

from torchvision import models, transforms
from continuum import ClassIncremental
from continuum.datasets import CIFAR100
from continuum.tasks import split_train_val
import torch
from tqdm import tqdm

import wandb

# wandb.init(project="tasksim", entity="omar-s")

parser = argparse.ArgumentParser(description='cnn_finetune cifar 10 example')
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
                    help='input batch size for testing (default: 64)')
parser.add_argument('--num-epochs', type=int, default=16, metavar='N',
                    help='number of epochs to train (default: 16)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')

args = parser.parse_args()

NUM_LABELS = 100
INCREMENT_SIZE = 10

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
## REMOVE LATER
torch.manual_seed(0)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )])

## Configure model
class PretrainedModel(torch.nn.Module):
    def __init__(self, init_head_size):
        super().__init__()
        self.encoder = models.resnet34(pretrained=True)
        self.head_size = 0
        self.fc_in_features = self.encoder.fc.in_features
        for param in self.encoder.parameters():
            param.requires_grad = False
        
    def extend_head(self, n):
        new_head_size = self.head_size + n
        new_head = torch.nn.Linear(self.fc_in_features, new_head_size)
        if self.head_size != 0: ## Save old class weights
            new_head.weight.data[:self.head_size] = self.encoder.fc.weight.data.clone().to(device)
            new_head.bias.data[:self.head_size] = self.encoder.fc.bias.data.clone().to(device)
        self.encoder.fc = new_head
        self.head_size = new_head_size
        self.encoder.to(device)
    
    def forward(self, x):
        return self.encoder(x)

def train(model, train_loader, optim: torch.optim.Optimizer, criterion: torch.nn.CrossEntropyLoss):
    model.train()
    for inputs, labels, tasks in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optim.zero_grad()
        outs = model(inputs)
        loss = criterion(outs, labels)
        loss.backward()
        optim.step()
    return

def evaluate(model, test_loader, criterion):
    model.eval()
    total_loss = 0
    accuracy = 0
    with torch.no_grad():
        for inputs, labels, tasks in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outs = model(inputs)
            accuracy += torch.sum(outs.max(1)[1] == labels).float() / len(labels)
            loss = criterion(outs, labels)
            total_loss += loss
    print(f"Average loss: {total_loss/len(test_loader)}")
    print(f"Average accuracy: {accuracy/len(test_loader)}")
    return

def run_train_loop(model, train_loader, test_loader, optim, lr_scheduler, criterion, num_epochs):
    for i in range(num_epochs):
        train(model, train_loader, optim, criterion)
        evaluate(model, test_loader, criterion)
        # lr_scheduler.step()

## Load data
dataset = CIFAR100('./data', train=True, download=True)
scenario = ClassIncremental(
    dataset,
    increment=10,
    initial_increment=0,
    transformations=[transform]
)
print(f"Number of classes: {scenario.nb_classes}.")
print(f"Number of tasks: {scenario.nb_tasks}.")


## Configure loss, optimizer, lr scheduler
model = PretrainedModel(INCREMENT_SIZE)
criterion = torch.nn.CrossEntropyLoss()

for task_id, train_taskset in tqdm(enumerate(scenario)):
    train_taskset, val_taskset = split_train_val(train_taskset, val_split=0.1)
    train_loader = torch.utils.data.DataLoader(
        train_taskset, batch_size=32, shuffle=True)
    val_loader = torch.utils.data.DataLoader(
        val_taskset, batch_size=32, shuffle=True)
    model.extend_head(train_taskset.nb_classes)
    # optim.add_param_group({"params" : model.encoder.fc.parameters()})
    optim = torch.optim.SGD(model.encoder.fc.parameters(), lr=args.lr, momentum=0.9)
    # Decay LR by a factor of 0.1 every 7 epochs
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=7, gamma=0.1)
    print(f"Model head size: {model.head_size}")
    # print(optim.param_groups)
    run_train_loop(model, train_loader, val_loader, optim, lr_scheduler, criterion, args.num_epochs)
