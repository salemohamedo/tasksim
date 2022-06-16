import pandas as pd
import numpy as np
import torch
import pickle
from pathlib import Path
from typing import Tuple
from dataclasses import asdict
from tqdm import tqdm
from argparse import ArgumentParser
import json

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = ArgumentParser()
parser.add_argument('--delete', action='store_true')
clean_args = parser.parse_args()
RUNS_DIR = Path('results')

def rmdir(directory):
    for item in directory.iterdir():
        if item.is_dir():
            rmdir(item)
        else:
            item.unlink()
    directory.rmdir()

corrupt_models = []
for run in RUNS_DIR.iterdir():
    try:
        with open(run / 'config.txt', 'r') as f:
            args = json.load(f)
        pd.read_csv(run / 'results.csv')
        if args['head_type'] == 'linear':
            n_tasks = args['n_tasks']
            if args['freeze_features'] == False: ## For e2e models we want to save model ckpts
                if n_tasks != len(list(head.glob('*ckpt*'))):
                    corrupt_models.append(str(run))
                    # corrupt_models.append(str(head).replace('linear', 'nmc'))
            # if 'init' not in setting.name.lower():
            #     emb = torch.load(head / 'embeddings.pt')
            #     assert len(emb) == n_tasks - 1
    except Exception as e:
        corrupt_models.append(str(run))

if len(corrupt_models) > 0:
    for c in corrupt_models:
        print('Corrupt:', c)
    print(f"# Corrupt models: {len(corrupt_models)}")
    if clean_args.delete == True:
        for c in corrupt_models:
            rmdir(Path(c))
        print("Corrupt models deleted!!!")
else:
    print("Found no corrupt models!!")