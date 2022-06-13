from pathlib import Path
from scipy.stats import pearsonr
import pickle
import pandas as pd
import numpy as np
import json
from similarity_metrics.task2vec import cosine
from dataclasses import dataclass
import matplotlib.pyplot as plt
from typing import List
plt.style.use('seaborn-whitegrid')

@dataclass
class ResultsSummary:
    acc_lin: float = None
    fgt_lin: float = None
    acc_nmc: float = None
    fgt_nmc: float = None
    sim_acc_lin: float = None
    sim_fgt_lin: float = None
    sim_acc_nmc: float = None
    sim_fgt_nmc: float = None

def parse_acc_forgetting(results_file: str) -> List:
    results = pd.read_csv(results_file)
    results = results.to_numpy()
    results = results[:,1:]
    mean_accs = []
    mean_fgts = []
    num_tasks = results.shape[0]
    for i in range(1, num_tasks):
        mean_acc = results[i][:i + 1].mean()
        mean_fgt = np.mean([results[j][j] - results[i][j] for j in range(i)])
        mean_accs.append(mean_acc)
        mean_fgts.append(mean_fgt)
    return mean_accs, mean_fgts

def parse_task_sim(embeddings_file: str) -> List:
    with open(embeddings_file, 'rb') as f:
        embeddings = pickle.load(f)
    running_distance = 0
    distances = []
    for i in range(1, len(embeddings)):
        running_distance += cosine(embeddings[i - 1], embeddings[i])
        distances.append(running_distance / i)
    return distances

def get_run_results(run_dir, nmc=True):
    run_dir = Path(run_dir)
    if not run_dir.exists():
        print(f"No Run Directory: {run_dir}")
    case_list = []
    for f in run_dir.iterdir():
        if str(f).find("case") > -1:
            case_list.append(f)
    id_list = [int(str(x).split("_")[-1].split(".")[0]) for x in case_list]
    num_cases = max(id_list) + 1
    print(f"\n## Total Number Cases: {num_cases}")
    lin_accs, lin_fgts = [], []
    if nmc:
        nmc_accs, nmc_fgts = [], []
    else:
        nmc_accs, nmc_fgts = None, None
    task_sims = []
    for case_id in range(num_cases):
        print(lin_accs)
        lin_acc, lin_fgt = parse_acc_forgetting(run_dir / f"case_{case_id}.acc.lin")
        if nmc:
            nmc_acc, nmc_fgt = parse_acc_forgetting(run_dir / f"case_{case_id}.acc.nmc")
            nmc_accs.extend(nmc_acc)
            nmc_fgts.extend(nmc_fgt)
        task_sim = parse_task_sim(run_dir / f'case_{case_id}.emb')
        lin_accs.extend(lin_acc)
        lin_fgts.extend(lin_fgt)
        task_sims.extend(task_sim)
    
    return lin_accs, lin_fgts, nmc_accs, nmc_fgts, task_sims

def process_run_results(lin_accs, lin_fgts, nmc_accs, nmc_fgts, task_sims) -> ResultsSummary:
    results = ResultsSummary()
    results.acc_lin = np.mean(lin_accs) * 100
    results.fgt_lin = np.mean(lin_fgts) * 100
    if nmc_accs is not None and nmc_fgts is not None:
        results.acc_nmc = np.mean(nmc_accs) * 100
        results.fgt_nmc = np.mean(nmc_fgts) * 100

    ## Invert distances so smaller is better
    invert_task_sims = np.array(task_sims) * -1

    results.sim_acc_lin = pearsonr(lin_accs, invert_task_sims)[0]
    results.sim_fgt_lin = pearsonr(lin_fgts, invert_task_sims)[0]
    if nmc_accs is not None and nmc_fgts is not None:
        results.sim_acc_nmc = pearsonr(nmc_accs, invert_task_sims)[0]
        results.sim_fgt_nmc = pearsonr(nmc_fgts, invert_task_sims)[0]
    return results

def plot_similarity_correlation(task_metric, task_sim, task_metric_label, title, out_file):
    assert len(task_metric) == len(task_sim)
    plt.scatter(task_sim, task_metric, marker='o')
    plt.xlabel("Task dissimilarity")
    plt.ylabel(task_metric_label)
    plt.title(title)
    plt.savefig(out_file)

def load_config(run_dir):
    run_dir = Path(run_dir)
    with open(run_dir / 'config.txt', 'r') as config_fp:
        return json.load(config_fp)
