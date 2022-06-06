from pathlib import Path
from scipy.stats import pearsonr
import pickle
import pandas as pd
import numpy as np
import json
from similarity_metrics.task2vec import cosine
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')

def parse_acc_forgetting(results_file):
    results = pd.read_csv(results_file)
    results = results.to_numpy()
    results = results[:,1:]
    acc = results[-1].mean()
    forgetting = 0
    for i in range(results.shape[1] - 1):
        forgetting += results[i][i] - results[-1][i]
    return acc, forgetting / (results.shape[1] - 1)

def parse_task_sim(embeddings_file):
    with open(embeddings_file, 'rb') as f:
        embeddings = pickle.load(f)
    total_distance = 0
    for i in range(len(embeddings) - 1):
        total_distance += cosine(embeddings[i], embeddings[i + 1])
    return total_distance

def get_run_results(run_dir):
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
    lin_accs = []
    lin_fgts = []
    nmc_accs = []
    nmc_fgts = []
    task_sims = []
    for case_id in range(num_cases):
        lin_acc, lin_fgt = parse_acc_forgetting(run_dir / f"case_{case_id}.acc.lin")
        nmc_acc, nmc_fgt = parse_acc_forgetting(run_dir / f"case_{case_id}.acc.nmc")
        task_sim = parse_task_sim(run_dir / f'case_{case_id}.emb')
        lin_accs.append(lin_acc)
        lin_fgts.append(lin_fgt)
        nmc_accs.append(nmc_acc)
        nmc_fgts.append(nmc_fgt)
        task_sims.append(task_sim)
    
    return lin_accs, lin_fgts, nmc_accs, nmc_fgts, task_sims

def process_run_results(lin_accs, lin_fgts, nmc_accs, nmc_fgts, task_sims):
    results = {}
    results['acc_lin'] = np.mean(lin_accs)
    results['fgt_lin'] = np.mean(lin_fgts)
    results['acc_nmc'] = np.mean(nmc_accs)
    results['fgt_nmc'] = np.mean(nmc_fgts)

    ## Invert distances so smaller is better
    invert_task_sims = np.array(task_sims) * -1

    results['sim_acc_lin'] = pearsonr(lin_accs, invert_task_sims)[0]
    results['sim_fgt_lin'] = pearsonr(lin_fgts, invert_task_sims)[0]
    results['sim_acc_nmc'] = pearsonr(nmc_accs, invert_task_sims)[0]
    results['sim_fgt_nmc'] = pearsonr(nmc_fgts, invert_task_sims)[0]
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
