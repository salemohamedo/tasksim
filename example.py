import time
import argparse

from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
import torch
import numpy, random
from tqdm import tqdm

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

args = parser.parse_args()
print(args)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )])
    
## Load data
train_data = datasets.CIFAR10(
    root='./data', train=True, transform=transform, download=True)
test_data = datasets.CIFAR10(
    root='./data', train=False, transform=transform, download=True)

NUM_LABELS = len(train_data.classes)

train_loader = DataLoader(
    train_data, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True, worker_init_fn=seed_worker,
    generator=g
)
test_loader = DataLoader(
    test_data, batch_size=args.test_batch_size, shuffle=False, num_workers=4, pin_memory=True, worker_init_fn=seed_worker,
    generator=g
)

## Configure model
model = models.resnet34(pretrained=True)
if args.freeze_features:
    for param in model.parameters():
        param.requires_grad = False
fc_features = model.fc.in_features
model.fc = torch.nn.Linear(fc_features, NUM_LABELS)
model.to(device)
optim_params = model.fc.parameters() if args.freeze_features else model.parameters()

## Configure loss, optimizer, lr scheduler
criterion = torch.nn.CrossEntropyLoss()
# optim = torch.optim.SGD(model.fc.parameters(), lr=args.lr, momentum=0.9)
# optim = torch.optim.Adam(model.fc.parameters(), lr=3e-4)
# # Decay LR by a factor of 0.1 every 7 epochs
# lr_scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=7, gamma=0.1)

optim = torch.optim.SGD(optim_params, lr=args.lr, momentum=0.9, weight_decay=5e-4)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.num_epochs)

# xs, ys = None, None
def train(model, train_loader, optim: torch.optim.Optimizer, criterion : torch.nn.CrossEntropyLoss):
    model.train()
    total_loss = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optim.zero_grad()
        outs = model(inputs)
        loss = criterion(outs, labels)
        # print(loss)
        total_loss += loss
        loss.backward()
        optim.step()
    # print("Train: ", total_loss / len(train_loader))
    return

def evaluate(model, test_loader, criterion):
    model.eval()
    total_loss = 0
    accuracy = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outs = model(inputs)
            loss = criterion(outs, labels)
            accuracy += torch.sum(outs.max(1)[1] == labels).float() / len(labels)
            total_loss += loss
    # print("Test: ", total_loss/len(test_loader))
    return total_loss/len(test_loader), accuracy/len(test_loader)

start = time.time()
for i in range(args.num_epochs):
    train(model, train_loader, optim, criterion)
    loss, acc = evaluate(model, test_loader, criterion)
    print(f"Epoch: {i}\tLoss: {loss:.4f}\tAcc: {acc*100:.2f}%\tLR: {lr_scheduler.get_last_lr()}")
    lr_scheduler.step()
print(f"Train time: {time.time()-start:.2f}s")
print(f"Total run time: {time.time()-very_start:.2f}s")
