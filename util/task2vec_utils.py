
from nngeometry.metrics import FIM, FIM_MonteCarlo
from models import TasksimModel, NMC_Classifier
from nngeometry.object import PMatDiag
import torch
from similarity_metrics.task2vec import Task2Vec, get_hessian, cosine, normalized_cosine
import math
import wandb
from torch.nn.functional import cosine_similarity
from pathlib import Path


def fit_classifier_nmc(model: TasksimModel, data_loader):
    for inputs, *labels in data_loader:
        labels = labels[0]
        inputs, labels = inputs.to(model.device), labels.to(model.device)
        model(inputs, labels)
        model.classifier.update_means(labels, 0)
    return

def fit_classifier(model: TasksimModel, data_loader, next_task: int, old: bool, epochs):
    learning_rate = 0.0004
    weight_decay = 0.0001
    optim = torch.optim.Adam(model.classifier.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = torch.nn.CrossEntropyLoss()
    model.train()
    print(f"FIM Classifier Size: {model.classifier.weight.shape[0]}")
    print(f"\n## Fitting FIM classifier ##\n")
    for _ in range(epochs):
        accuracy = 0
        for inputs, *labels in data_loader:
            labels = labels[0]
            inputs, labels = inputs.to(model.device), labels.to(model.device)
            optim.zero_grad()
            outs = model(inputs, labels)
            loss = criterion(outs, labels)
            accuracy += torch.sum(outs.max(1)
                                  [1] == labels).float() / len(labels)
            loss.backward()
            optim.step()
        accuracy = accuracy / len(data_loader)
        if wandb.run is not None:
            wandb.log({f'Task2Vec_Task_{next_task}_{"prev" if old else "next"}_train_acc': accuracy})
        print(f"FIM Classifier Train Accuracy: {accuracy:.2f}")

def task2vec(
    model: TasksimModel, 
    old_task_dataloader, 
    new_task_dataloader, 
    n_new_classes, 
    next_task, 
    type, 
    epochs, 
    combined_head, 
    old_task_test_loader, 
    new_task_test_loader
):
    n_old_classes = model.head_size
    cur_head = model.classifier
    print(f'###\tComputing Task2Vec embedding of previous tasks\t###\n')
    old_tasks_vec = compute_task2vec(
        model, 
        old_task_dataloader, 
        n_old_classes, 
        type, 
        next_task, 
        old=True, 
        epochs=epochs,
        test_loader=old_task_test_loader)
    print(f'\n###\tComputing Task2Vec embedding of new task\t###')
    new_head = None
    if combined_head:
        new_head = torch.nn.Linear(
            model.fc_in_features, n_old_classes + n_new_classes, device=model.device)
        new_head.weight.data[:n_old_classes] = cur_head.weight.data.clone()
        new_head.bias.data[:n_old_classes] = cur_head.bias.data.clone()
    new_task_vec = compute_task2vec(
        model, 
        new_task_dataloader, 
        n_new_classes, 
        type, 
        next_task, 
        old=False, 
        epochs=epochs, 
        head=new_head,
        test_loader=new_task_test_loader)
    model.classifier = cur_head
    return old_tasks_vec, new_task_vec


def compute_task2vec(model: TasksimModel, dataloader, num_classes, type, next_task: int, old: bool, epochs, head=None, test_loader=None):
    if type == 'prototype':
        model.classifier = NMC_Classifier(model.fc_in_features, model.device)
        model.classifier.extend_head(num_classes)
        fit_classifier_nmc(model, dataloader)
        nmc_classifier = model.classifier
        model.classifier = torch.nn.Linear(model.fc_in_features, num_classes, device=model.device, bias=False)
        model.classifier.weight.data = nmc_classifier.class_means
    else:
        if head:
            model.classifier = head
        else:
            model.classifier = torch.nn.Linear(
                model.fc_in_features, num_classes, device=model.device)
        fit_classifier(model, dataloader, next_task, old, epochs)
    model.eval()

    if test_loader:
        fisher_loader = test_loader
    else:
        fisher_loader = dataloader

    if model.frozen_features:
        model.unfreeze_features()
        embedding = compute_fisher(model, fisher_loader)
        model.freeze_features()
    else:
        embedding = compute_fisher(model, fisher_loader)
    return embedding

def compute_fisher(model: TasksimModel, dataloader):  
    fim_diag: PMatDiag = FIM_MonteCarlo(model, dataloader, PMatDiag,
                             trials=1, device=model.device)
    config_path = Path(f'./config/{model.model_name}.json')
    if not config_path.exists():
        import json
        with open(config_path, 'w') as fp:
            json.dump(fim_diag.generator.layer_collection.p_pos, fp)
    return fim_diag.get_diag()
    # # return torch.Tensor(get_hessian(Task2Vec(model).embed2(dataloader.dataset)))
    # task2vec = Task2Vec(model).embed2(dataloader.dataset)
    # print('here')

def cos_similarity(old, new, norm=False):
    if norm:
        old = old / (old + new + 1e-8)
        new = new / (old + new + 1e-8)
    return cosine_similarity(old, new, dim=0).cpu()
