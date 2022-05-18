import time
import argparse

from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm

torch.manual_seed(0)

parser = argparse.ArgumentParser(description='Tasksim Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
                    help='input batch size for testing (default: 64)')
parser.add_argument('--num-epochs', type=int, default=1000, metavar='N',
                    help='number of epochs to train (default: 64)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')

args = parser.parse_args()

# NUM_LABELS = 100

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
# train_data.data = train_data.data[:args.batch_size*2]
# test_data.data = test_data.data[:args.test_batch_size*2]

train_loader = DataLoader(
    train_data, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True
)
test_loader = DataLoader(
    test_data, batch_size=args.test_batch_size, shuffle=False, num_workers=2, pin_memory=True
)

# inputs, labels = next(iter(train_loader))
# inputs, labels = inputs.to(device), labels.to(device)

## Configure model
model = models.resnet34(pretrained=True)
# for param in model.parameters():
#     param.requires_grad = False
fc_features = model.fc.in_features
model.fc = torch.nn.Linear(fc_features, NUM_LABELS)
model.to(device)

## Configure loss, optimizer, lr scheduler
criterion = torch.nn.CrossEntropyLoss()
# optim = torch.optim.SGD(model.fc.parameters(), lr=args.lr, momentum=0.9)
# optim = torch.optim.Adam(model.parameters(), lr=3e-4)
# # Decay LR by a factor of 0.1 every 7 epochs
# lr_scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=7, gamma=0.1)

optim = torch.optim.SGD(model.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
# lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.num_epochs)

# xs, ys = None, None
def train(model, train_loader, optim: torch.optim.Optimizer, criterion : torch.nn.CrossEntropyLoss):
    model.train()
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optim.zero_grad()
        outs = model(inputs)
        loss = criterion(outs, labels)
        # print(loss)
        loss.backward()
        optim.step()
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
    return total_loss/len(test_loader), accuracy/len(test_loader)

start = time.time()
for i in range(args.num_epochs):
    train(model, train_loader, optim, criterion)
    loss, acc = evaluate(model, test_loader, criterion)
    print(f"Epoch: {i}\tLoss: {loss:.4f}\tAcc: {acc*100:.2f}%\tLR: {None}")
    # lr_scheduler.step()
print(time.time()-start)