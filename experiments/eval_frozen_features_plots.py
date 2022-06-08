from utils.eval_utils import get_run_results, plot_similarity_correlation, load_config

from collections import OrderedDict
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')

RESULT_DIRS = [
    ("CIFAR-10",'results/run_013'),
    ("CIFAR-100", 'results/run_014'),
    ("MNIST", 'results/run_015'),
    ("CUB200", 'results/run_016')
]

DATASET_NAMES = OrderedDict(RESULT_DIRS).keys()
SIM_METRICS = ["Task2Vec"]

sim_results = []
task_metric_results = []

for run in RESULT_DIRS:
    dataset = run[0]
    run_dir = run[1]
    run_config = load_config(run_dir)
    assert dataset.lower() == run_config["dataset"].lower()

    lin_accs, lin_fgts, nmc_accs, nmc_fgts, task_sims = get_run_results(
        run_dir)
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    fig.suptitle(dataset)
    ax1.scatter(task_sims, 1 - np.array(lin_accs))
    ax1.set_aspect('equal')
    ax1.set_xlabel("Task dissimilarity")
    ax2.scatter(task_sims, lin_fgts)
    ax2.set_xlabel("Task dissimilarity")
    ax2.set_aspect('equal')
    ax3.scatter(task_sims, 1 - np.array(nmc_accs))
    ax3.set_xlabel("Task dissimilarity")
    ax3.set_aspect('equal')
    fig.savefig("test.png")
    break


# def plot_similarity_correlation(task_metric, task_sim, task_metric_label, title, out_file):
#     assert len(task_metric) == len(task_sim)
#     plt.scatter(task_sim, task_metric, marker='o')
#     plt.xlabel("Task dissimilarity")
#     plt.ylabel(task_metric_label)
#     plt.title(title)
#     plt.savefig(out_file)
