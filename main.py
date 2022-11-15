from pathlib import Path
from dataclasses import asdict
import time

from continuum import ClassIncremental, rehearsal, ContinualScenario
from continuum.tasks import split_train_val
import torch
from torch.utils.data import ConcatDataset, TensorDataset
import numpy as np
import math
import pandas as pd
from copy import deepcopy

from util.dataset_utils import DATASETS, load_dataset, get_transform, get_dataset_class_names
from util.dataset_classes import CIFAR10_taxonomy
from models import TasksimModel, get_optimizer_lr_scheduler, PRETRAINED_MODELS, TASK2VEC_IGNORE_MODELS
from util.utils import get_model_state_dict, get_full_results_dir, set_seed, save_results, save_model
from util.task2vec_utils import task2vec, cos_similarity
from util.tasksim_args import TaskSimArgs, parse_args
from util.eval_utils import evaluate_results
from util.test_perturbations import TinyDomainIncScenario
from metrics import compute_metrics

import wandb

DOMAIN_INC_DISTS = ["gaussian_noise", "shot_noise", "impulse_noise", "speckle_noise", "gaussian_blur",
                    "defocus_blur", "motion_blur", "zoom_blur", "fog", "snow", "spatter", "contrast",
                    "brightness", "saturate", "elastic_transform", "glass_blur"]

def train(model: TasksimModel, train_loader, optim: torch.optim.Optimizer, criterion: torch.nn.CrossEntropyLoss, epoch, optim_type, task_id):
    model.train()
    total_loss = 0.
    accuracy = 0
    for i, (inputs, *labels) in enumerate(train_loader):
        labels = labels[0]
        inputs, labels = inputs.to(model.device), labels.to(model.device)
        if not model.nmc:
            optim.zero_grad()
        outs = model(inputs, labels)
        if model.nmc:
            model.classifier.update_means(labels.detach(), epoch)
        loss = criterion(outs, labels)
        total_loss += float(loss)
        if not model.nmc:
            loss.backward()
            # if optim_type == 'adam':
            #     torch.nn.utils.clip_grad_norm(model.parameters(), 1)
            optim.step()
        accuracy += torch.sum(outs.max(1)
                                  [1] == labels).float() / len(labels)
        total_loss += float(loss)
    return total_loss/len(train_loader), accuracy/len(train_loader)

def evaluate(model: TasksimModel, test_loader, criterion):
    model.eval()
    total_loss = 0.
    accuracy = 0
    with torch.no_grad():
        for inputs, *labels in test_loader:
            labels = labels[0]
            inputs, labels = inputs.to(model.device), labels.to(model.device)
            outs = model(inputs)
            loss = criterion(outs, labels)
            accuracy += torch.sum(outs.max(1)
                                  [1] == labels).float() / len(labels)
            total_loss += float(loss)
    return total_loss/len(test_loader), accuracy/len(test_loader)

def encode_features(model, data_loader):
    x, y = [], []
    for inputs, *labels in data_loader:
        labels = labels[0]
        inputs = inputs.to(model.device)
        features = model.encode_features(inputs)
        x.append(features.data.cpu().clone())
        y.append(labels.clone())
    return TensorDataset(torch.concat(x), torch.concat(y))

def run_train_loop(args: TaskSimArgs, model: TasksimModel, train_loader, val_loader, optim, lr_scheduler: torch.optim.lr_scheduler.StepLR, criterion, num_epochs, task_id):
    print(len(train_loader.dataset), len(val_loader.dataset))
    if args.head_type == 'nmc' and not args.freeze_features:
        model_ckpt = get_model_state_dict(args, task_id)
        assert model_ckpt is not None
        model.feature_extractor.load_state_dict(model_ckpt)
        # print('loaded')

    best_val_loss = float('inf')
    best_model = None
    bad_iters = 0
    if model.nmc:
        num_epochs = 1
    if model.frozen_features: ## Pre-encode features and only train classifier
        train_encoded_dataset = encode_features(model, train_loader)
        train_loader = torch.utils.data.DataLoader(
            train_encoded_dataset, batch_size=args.batch_size, shuffle=True,
             num_workers=4, drop_last=False, pin_memory=True)
    for i in range(num_epochs):
        train_loss, train_acc = train(model, train_loader, optim, criterion, i, args.optim, task_id=task_id) 
        val_loss, val_acc = evaluate(model, val_loader, criterion)
        print(f'Epoch {i}: [Train] Acc: {train_acc:.2f} Loss: {train_loss:.2f} \
        [Val] Acc: {val_acc: .2f} Loss: {val_loss: .2f}')
        if args.wandb:
            wandb.log({
                f'Task_{task_id}_val_acc': val_acc,
                f'Task_{task_id}_val_loss': val_loss,
                f'Task_{task_id}_train_acc': train_acc,
                f'Task_{task_id}_train_loss': train_loss
            })
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            bad_iters = 0
            if args.wandb:
                wandb.run.summary[f'Task_{task_id}_best_val_acc'] = val_acc
                wandb.run.summary[f'Task_{task_id}_best_val_loss'] = val_loss
            best_model = deepcopy(model.state_dict())
        else:
            bad_iters += 1
            if bad_iters == args.patience:
                break

        # if not model.nmc:
        #     lr_scheduler.step()
    model.load_state_dict(best_model)
    if args.save_feature_extractor and args.head_type == 'linear':
        save_model(args, model.feature_extractor.state_dict(), task_id)


def run_cl_sequence(args: TaskSimArgs, model: TasksimModel, train_scenario: ClassIncremental, test_scenario: ClassIncremental, replay_buff: rehearsal.RehearsalMemory = None):
    ## Configure loss, optimizer, lr scheduler
    criterion = torch.nn.CrossEntropyLoss()
    prev_test_loaders = []
    old_train_loader = None
    embeddings = None
    sim_metrics = []
    results = np.zeros((args.n_tasks, args.n_tasks))
    task2vec_buffer = None
    cumulative_val_dataset = None
    if args.init:
        init_fe_weights = deepcopy(model.feature_extractor.state_dict())
    lr = args.lr
    for task_id, (train_taskset, test_taskset) in enumerate(zip(train_scenario, test_scenario)):
        train_taskset._y = train_taskset._y.astype('int64')
        test_taskset._y = test_taskset._y.astype('int64')
        if task_id >= args.n_tasks:
            break
        if args.mixup and task_id == 1:
            train_taskset = mixup_task(train_scenario, args.mixup_lambda, args.mixup_type)
            test_taskset = mixup_task(test_scenario, args.mixup_lambda, args.mixup_type)

        if args.domain_inc and task_id > 0:
            num_new_classes = 0
        else:
            num_new_classes = train_taskset.nb_classes
        # print(set(train_taskset._y))
        print(f"\n######\t Learning Task {task_id} [{num_new_classes} Classes]\t######\n")

        ## Load data
        train_taskset, val_taskset = split_train_val(train_taskset, val_split=0.1)
        test_loader = torch.utils.data.DataLoader(
            test_taskset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=False, pin_memory=True)

        if args.metrics and task_id > 0:
            ## This train loader only contains data from unseen classes (no replay buffer samples), used for computing metrics only!
            new_task_train_loader = torch.utils.data.DataLoader(
                train_taskset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=False, pin_memory=True)
            sim_metrics.append(compute_metrics(model, old_train_loader, new_task_train_loader))

        # Compute task2vec embedding
        if args.task2vec and task_id > 0 and args.model not in TASK2VEC_IGNORE_MODELS:
            model.set_task2vec_mode(True)
            original_new_task_offset = train_taskset._y.min()
            train_taskset._y -= original_new_task_offset

            new_task_dataloader = torch.utils.data.DataLoader(
                train_taskset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=False, pin_memory=True)
            old_tasks_dataloader = torch.utils.data.DataLoader(
            task2vec_buffer, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=False, pin_memory=True)
            emb = {}
            if args.task2vec_fisher_with_test:
                old_task_test_loader = prev_test_loaders[-1]
                new_task_test_loader = test_loader
            else:
                old_task_test_loader = None
                new_task_test_loader = None
            for type in ['linear']:
                old_vec, new_vec = task2vec(
                    model=model, 
                    old_task_dataloader=old_tasks_dataloader,
                    new_task_dataloader=new_task_dataloader, 
                    n_new_classes=num_new_classes,
                    next_task=task_id, 
                    type=type, 
                    epochs=args.task2vec_epochs, 
                    combined_head=args.task2vec_combined_head,
                    old_task_test_loader=old_task_test_loader,
                    new_task_test_loader=new_task_test_loader)
                emb[f'{type}_old_vec'] = old_vec
                emb[f'{type}_new_vec'] = new_vec
            if not embeddings:
                embeddings = []
            embeddings.append(emb)

            train_taskset._y += original_new_task_offset
            model.set_task2vec_mode(False)

        if cumulative_val_dataset == None:  # First task
            cumulative_val_dataset = ConcatDataset([val_taskset])
        else:
            cumulative_val_dataset = ConcatDataset(
                cumulative_val_dataset.datasets + [val_taskset])

        if args.val_all:
            val_dataset = cumulative_val_dataset
        else:
            val_dataset = val_taskset

        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=False, pin_memory=True)

        ## Add replay buffer to training data
        if replay_buff is not None and task_id > 0:
            mem_x, mem_y, mem_t = replay_buff.get()
            train_taskset.add_samples(mem_x, mem_y, mem_t)

        train_loader = torch.utils.data.DataLoader(
            train_taskset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=False, pin_memory=True)

        train_ys = set(train_taskset._y)
        test_ys = set(test_taskset._y)
        # assert train_ys == test_ys

        ## Update model, optimizer
        if not args.domain_inc or task_id == 0:
            model.extend_head(num_new_classes)

        if args.init and task_id > 0:
            model.feature_extractor.load_state_dict(init_fe_weights)
            model.classifier.init_weights()
            model.classifier.to(model.device)

        optim, lr_scheduler = None, None
        if args.head_type == 'linear':
            if args.freeze_features:
                optim_params = model.classifier.parameters()
            else:
                optim_params = model.parameters()
            # if task_id in [1, 3]:
            #     lr /= 10
            print(f'Learning rate: {lr}')
            optim, lr_scheduler = get_optimizer_lr_scheduler(args.optim, optim_params, lr, args.momentum)

        ## Train and evaluate test accuracy on current task
        print("\n###\tTraining\t###\n")
        print(f"Model head size: {model.classifier.weight.shape[0]}")
        run_train_loop(args, model, train_loader, val_loader, optim,
                       lr_scheduler, criterion, args.num_epochs, task_id)
        loss, acc = evaluate(model, test_loader, criterion)
        results[task_id][task_id] = acc

        print(f"\nTask {task_id} IID Test Accuracy: {acc*100:.2f}")

        ## Eval test accuracy on previous tasks
        if task_id > 0:
            print(f"\n## Testing Previous Tasks ##\n")
            for i, tl in enumerate(prev_test_loaders):
                loss, acc = evaluate(model, tl, criterion)
                print(f"Task: {i}\tLoss: {loss:.4f}\tAcc: {acc*100:.2f}%")
                results[task_id][i] = acc
        prev_test_loaders.append(test_loader)
        
        if task2vec_buffer == None:  # First task
            task2vec_buffer = ConcatDataset([train_taskset])
        else:
            task2vec_buffer = ConcatDataset(task2vec_buffer.datasets + [train_taskset])

        ## Update replay buffer
        if replay_buff is not None:
            replay_buff.add(*train_scenario[task_id].get_raw_samples(), z=None)

        old_train_loader = train_loader
    return results, embeddings, sim_metrics

def prepare_scenario(dataset, n_tasks, n_classes_per_task, transform, order=None, domain_inc=False):
    if domain_inc:
        return ContinualScenario(dataset, transformations=transform)
    num_classes = dataset.num_classes
    if n_classes_per_task == None:
        if num_classes % n_tasks == 0:
            incs = [(num_classes / n_tasks)] * n_tasks
        else:
            incs = [math.ceil(num_classes / n_tasks)] * (n_tasks - 1)
            incs.append(num_classes - sum(incs))
    else:
        assert n_classes_per_task * n_tasks <= num_classes
        tasks = int(num_classes / n_classes_per_task)
        if num_classes % n_classes_per_task == 0:
            incs = [n_classes_per_task] * tasks
        else:
            incs = [n_classes_per_task] * math.ceil(tasks - 1)
            incs.append(num_classes - sum(incs))
    assert sum(incs) == num_classes
    return ClassIncremental(
        dataset,
        increment=incs,
        transformations=transform,
        class_order=order
    )

def mixup_task(scenario, mixup_coeff, mixup_type):
    assert len(scenario) == 2
    task1, task2 = list(scenario)
    min_task2_y = np.min(task2._y)
    task2._y -= min_task2_y
    ## To simplify mixup, make sure task1/task2 have same number of classes and same number of examples per class
    assert list(np.bincount(task1._y)) == list(np.bincount(task2._y))
    if mixup_type == 'noise':
        mixup_coeff = 1 - mixup_coeff
    for i in range(task1.nb_classes):
        task1_i_x = task1._x[task1._y == i]
        if mixup_type == 'task_data':
            mixup_data = task2._x[task2._y == i]
        elif mixup_type == 'noise':
            mixup_data = np.random.standard_normal(size=task1_i_x.shape)
        task2._x[task2._y == i] = mixup_coeff*task1_i_x + (1 - mixup_coeff)*mixup_data
    # task2._y += min_task2_y
    return task2

def run(args: TaskSimArgs):
    start_time = time.time()

    if args.wandb:
        wandb.init(project="CL-Similarity", entity="clip_cl",
                   config=args, reinit=True)
        args = TaskSimArgs(**wandb.run.config)

        wandb.run.name = f'{args.get_run_id()}_{wandb.run.id}'

    ## REMOVE ME
    args.batch_size = 64
    args.num_epochs = 20
    args.task2vec_epochs = 5
    if args.seed > 2:
        return
    ## REMOVE ME
    print(args)
    args.validate_args()

    if args.save_results and get_full_results_dir(args).exists():
        print(f'Specified results directory: {get_full_results_dir(args)} already exists. Exiting...')
        return

    set_seed(args.seed)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_dataset, test_dataset = load_dataset(args.dataset, args.domain_inc, args.n_classes_per_task)

    if args.model not in PRETRAINED_MODELS:
        transform = get_transform(args.dataset, args.model)

    class_order = [i for i in range(train_dataset.num_classes)]
    rng = np.random.RandomState(seed=args.seed)
    rng.shuffle(class_order)
    # print(get_dataset_class_names(args.dataset, class_order))

    model = TasksimModel(args.model, device, freeze_features=args.freeze_features,
                            multihead=args.multihead, pretrained=args.pretrained, 
                            nmc=args.head_type=='nmc', no_masking=args.no_masking).to(device)
    
    if args.model in PRETRAINED_MODELS:
        transform = model.transform
    
    if args.dataset == 'tiny' and args.domain_inc:
        dists = ["None"] + rng.choice(DOMAIN_INC_DISTS,
                                      replace=False, size=args.n_tasks-1).tolist()
        train_scenario = TinyDomainIncScenario(
            train_dataset, list_perturbation=dists, list_severity=[1])
        test_scenario = TinyDomainIncScenario(
            test_dataset, list_perturbation=dists, list_severity=[1])
        print(dists)
        if args.wandb:
            wandb.log({'distortions': str(dists)})
    else:
        rng.shuffle(class_order)
        train_scenario = prepare_scenario(
            train_dataset, args.n_tasks, args.n_classes_per_task, transform, class_order, args.domain_inc)
        test_scenario = prepare_scenario(
            test_dataset, args.n_tasks, args.n_classes_per_task, transform, class_order, args.domain_inc)

    print(f"Total number of classes: {train_scenario.nb_classes}.")
    print(f"Number of tasks: {args.n_tasks}.")
    print(class_order)

    replay_buff = None
    if args.replay_size_per_class != 0:
        if args.replay_size_per_class == -1:
            from collections import Counter
            if args.dataset == 'tiny' and args.domain_inc:
                ts = train_dataset.data[1]
            else:
                ts = train_dataset.dataset.targets
            replay_size_per_class = max(
                Counter(ts).values())
            print(replay_size_per_class)
        else:
            replay_size_per_class = args.replay_size_per_class
        replay_buff = rehearsal.RehearsalMemory(
            memory_size=replay_size_per_class * train_scenario.nb_classes,
            herding_method="random",
            fixed_memory=True,
            nb_total_classes=train_scenario.nb_classes
        )
    results, embeddings, sim_metrics = run_cl_sequence(args, model, train_scenario, test_scenario, replay_buff)
    summary = evaluate_results(args, results, embeddings, sim_metrics)
    results = pd.DataFrame(results)
    print(results)
    print(summary)

    print(f'\n#####\tPerformance summary\t#####\n')
    print(f'Final CL Accuracy: {summary["cl_acc"]*100:.2f}%')
    print(f'Mean IID Accuracy: {summary["mean_iid_acc"]*100:.2f}%')
    print(f'Mean Forgetting: {summary["mean_fgt"]*100:.2f}%')

    if args.wandb:
        wandb.log(summary)

    if args.save_results:
        save_results(args, results, embeddings)

    if args.wandb:
        wandb.run.finish()
    print(f'\nTotal Time Elapsed: {time.time()-start_time:.2f}')


if __name__ == '__main__':
    args = parse_args()
    # args.dataset = 'cifar-100'
    # args.model = 'efficient_net_nosy_teacher'
    # args.n_classes_per_task = 2
    # args.n_tasks = 2
    # # # args.batch_size = 20
    # # # args.domain_inc = True
    # args.replay_size_per_class = 0
    # args.num_epochs = 1
    # args.metrics = True
    # args.freeze_features = True
    # # # args.wandb = True
    # args.task2vec = True
    # args.task2vec_epochs = 1
    # # args.task2vec_combined_head = True
    # # torch.autograd.set_detect_anomaly(True)
    run(args)


