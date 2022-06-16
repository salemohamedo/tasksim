from pathlib import Path
import pickle
import pandas as pd
import numpy as np
import json
import torch
import random
import torch
from utils.tasksim_args import TaskSimArgs

BASE_RESULTS_PATH = './results'

def get_full_results_dir(args: TaskSimArgs):
    run_id = args.get_run_id()
    if args.results_dir == None:
        results_dir = Path(run_id)
    else:
        results_dir = Path(args.results_dir) / run_id
    return Path(BASE_RESULTS_PATH) / results_dir

def get_model_state_dict(args, task_id): 
    results_dir = get_full_results_dir(args)
    path = results_dir / f'fe_ckpt_task_{task_id}.pt'
    path = Path(str(path).replace('nmc', 'linear'))
    if path.exists():
        return torch.load(path)
    else:
        return None

def save_model(args, state_dict, task_id):
    results_dir = get_full_results_dir(args)
    if not results_dir.exists():
        results_dir.mkdir(parents=True)
    torch.save(state_dict, results_dir / f'fe_ckpt_task_{task_id}.pt')

def save_results(args: TaskSimArgs, results, embeddings):
    results_dir = get_full_results_dir(args)
    if not results_dir.exists():
        results_dir.mkdir(parents=True)

    with open(results_dir / 'config.txt', 'w') as config:
        json.dump(vars(args), config, indent=2)
    if results is not None:
        results.to_csv(results_dir / 'results.csv', float_format='%.5f')
    if args.save_embeddings and embeddings is not None and len(embeddings) > 0:
        torch.save(embeddings, results_dir / 'embeddings.pt')



def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)