from torchvision import models, transforms, datasets
import torch

NUM_LABELS = 100
BATCH_SIZE = 32
NUM_EPOCHS = 16
LEARNING_RATE = 0.001
STEP_SIZE = 7

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
    train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=2
)
test_loader = torch.utils.data.DataLoader(
    test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=2
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
optim = torch.optim.SGD(model.fc.parameters(), lr=LEARNING_RATE, momentum=0.9)
# Decay LR by a factor of 0.1 every 7 epochs
lr_scheduler = torch.lr_scheduler.StepLR(optim, step_size=STEP_SIZE, gamma=0.1)

def train(model, train_data, optim: torch.optim.Optimizer, lr_scheduler):
    for inputs, labels in train_data:
        inputs.to(device)
        labels.to(device)
        optim.zero_grad()
    return

def evaluate(model, test_data):
    return

for i in range(NUM_EPOCHS):
    lr_scheduler.step()
    train(model, train_data, optim, lr_scheduler)
    evaluate(model, test_data)