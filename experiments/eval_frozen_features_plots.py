from matplotlib.figure import Figure
from utils.eval_utils import get_run_results, plot_similarity_correlation, load_config

from collections import OrderedDict
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')

FROZEN_RESULT_DIRS = [
    ("CIFAR-10",'results/run_013'),
    ("CIFAR-100", 'results/run_014'),
    ("MNIST", 'results/run_015'),
    ("CUB200", 'results/run_016')
]

MULTIHEAD_RESULT_DIRS = [
    ("CIFAR-10",'results/run_022'),
    ("CIFAR-100", 'results/run_023'),
]

DATASET_NAMES = OrderedDict(MULTIHEAD_RESULT_DIRS).keys()
SIM_METRICS = ["Task2Vec"]

sim_results = []
task_metric_results = []

for run in MULTIHEAD_RESULT_DIRS:
    dataset = run[0]
    run_dir = run[1]
    run_config = load_config(run_dir)
    assert dataset.lower() == run_config["dataset"].lower()

    lin_accs, lin_fgts, nmc_accs, nmc_fgts, task_sims = get_run_results(run_dir, nmc=False)
    print(len(lin_accs))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle(dataset)
    ax1.scatter(task_sims, 1 - np.array(lin_accs))
    ax1.set_xlabel("Task dissimilarity")
    ax1.set_ylabel("Mean error rate")
    ax2.scatter(task_sims, lin_fgts)
    ax2.set_xlabel("Task dissimilarity")
    ax1.set_ylabel("Mean forgetting")
    fig.savefig("test.png")
    break


# def plot_similarity_correlation(task_metric, task_sim, task_metric_label, title, out_file):
#     assert len(task_metric) == len(task_sim)
#     plt.scatter(task_sim, task_metric, marker='o')
#     plt.xlabel("Task dissimilarity")
#     plt.ylabel(task_metric_label)
#     plt.title(title)
#     plt.savefig(out_file)
