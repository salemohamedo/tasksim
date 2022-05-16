import argparse

from torchvision import models, transforms, datasets
import torch
from tqdm import tqdm

parser = argparse.ArgumentParser(description='cnn_finetune cifar 10 example')
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
                    help='input batch size for testing (default: 64)')
parser.add_argument('--num-epochs', type=int, default=64, metavar='N',
                    help='number of epochs to train (default: 64)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')

args = parser.parse_args()

NUM_LABELS = 100

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )])

## Load data
train_data = datasets.CIFAR100(root='./data', train=True, transform=transform, download=True)
test_data = datasets.CIFAR100(root='./data', train=False, transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(
    train_data, batch_size=args.batch_size, shuffle=True, num_workers=2
)
test_loader = torch.utils.data.DataLoader(
    test_data, batch_size=args.test_batch_size, shuffle=False, num_workers=2
)

## Configure model
model = models.resnet34(pretrained=True)
for param in model.parameters():
    param.requires_grad = False
fc_features = model.fc.in_features
model.fc = torch.nn.Linear(fc_features, NUM_LABELS)
model.to(device)

## Configure loss, optimizer, lr scheduler
criterion = torch.nn.CrossEntropyLoss()
optim = torch.optim.SGD(model.fc.parameters(), lr=args.lr, momentum=0.9)
# Decay LR by a factor of 0.1 every 7 epochs
lr_scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=7, gamma=0.1)

def train(model, train_loader, optim: torch.optim.Optimizer, criterion : torch.nn.CrossEntropyLoss):
    model.train()
    for inputs, labels in tqdm(train_loader):
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
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outs = model(inputs)
            loss = criterion(outs, labels)
            total_loss += loss
    print(f"Average loss: {total_loss/len(test_loader.dataset)}")
    return

for i in tqdm(range(args.num_epochs)):
    train(model, train_loader, optim, criterion)
    evaluate(model, test_loader, criterion)
    lr_scheduler.step()