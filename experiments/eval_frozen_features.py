from utils.eval_utils import get_run_results, process_run_results, load_config

from collections import OrderedDict
import pandas as pd
import numpy as np

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

    run_results = get_run_results(run_dir)
    sum_results = process_run_results(*run_results)

    sim_results += [sum_results.sim_acc_lin, sum_results.sim_fgt_lin, sum_results.sim_acc_nmc]
    task_metric_result = [sum_results.acc_lin, sum_results.fgt_lin, sum_results.acc_nmc, sum_results.fgt_nmc]
    task_metric_results.append(task_metric_result)

sim_results = pd.DataFrame(sim_results, columns=SIM_METRICS).T
# sim_results.set_index(["SIM_METRICS"])
sim_results.columns = pd.MultiIndex.from_product(
    [DATASET_NAMES, ['$Acc_{LC}$', '$Forget_{LC}$', '$Acc_{NMC}$']])
task_metric_results = pd.DataFrame(
    task_metric_results, 
    columns=['$Acc_{LC}$', '$Forget_{LC}$', '$Acc_{NMC}$',  '$Forget_{NMC}$'],
    index=DATASET_NAMES)

# sim_results.to_csv('experiments/results_frozen_features.sim', float_format='%.3f')
sim_results.to_latex('figures/frozen-features-sim-metrics.tex', float_format='%.3f', escape=False, multicolumn=True, multicolumn_format='c')

# task_metric_results.to_csv('experiments/results_frozen_features.task', float_format='%.2f')
task_metric_results.to_latex(
    'figures/frozen-features-task-metrics.tex', float_format='%.2f', escape=False)
print(sim_results)
print(task_metric_results)
