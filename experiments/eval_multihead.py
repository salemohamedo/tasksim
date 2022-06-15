import datasets
from utils.eval_utils import get_run_results, process_run_results, load_config

from collections import OrderedDict
import pandas as pd
import numpy as np

nmc = False
BASE_DIR = 'results/multihead'
DATASETS = ['CIFAR_10', 'CIFAR_100', 'CUB200']
MODELS = ['ResNet', 'DenseNet', 'VGG']
# FROZEN_RESULT_DIRS = [
#     ("CIFAR-10",'results/old_results/run_013'),
#     ("CIFAR-100", 'results/old_results/run_014'),
#     ("MNIST", 'results/old_results/run_015'),
#     ("CUB200", 'results/old_results/run_016')
# ]

# MULTIHEAD_RESULT_DIRS = [
#     ("CIFAR-10", 'results/old_results/run_022'),
#     ("CIFAR-100", 'results/old_results/run_023'),
#     ("MNIST", 'results/old_results/run_024'),
#     ("CUB200", 'results/old_results/run_025')
# ]

# DATASET_NAMES = OrderedDict(MULTIHEAD_RESULT_DIRS).keys()

all_sim_results = []
# task_metric_results = []

for model in MODELS:
    model_sim_results = []
    for dataset in DATASETS:
        results_dir = f'{BASE_DIR}/{model.lower()}/{dataset.lower()}'
        run_config = load_config(results_dir)
        # assert dataset.lower().split('-_') == run_config["dataset"].lower().split('-_')

        run_results = get_run_results(results_dir, nmc=nmc)
        sum_results = process_run_results(*run_results)

        model_sim_results += [sum_results.sim_acc_lin, sum_results.sim_fgt_lin]
        task_metric_result = [sum_results.acc_lin, sum_results.fgt_lin]

        if nmc:
            model_sim_results.append(sum_results.sim_acc_nmc)
            task_metric_result += [sum_results.acc_nmc, sum_results.fgt_nmc]
        # task_metric_results.append(task_metric_result)
    all_sim_results.append(model_sim_results)
sim_results = pd.DataFrame(all_sim_results, index=MODELS)
print(sim_results)
# sim_results.set_index()
sim_result_cols = ['$Acc_{LC}$', '$Forget_{LC}$']
if nmc:
    sim_result_cols += ['$Acc_{NMC}$']
sim_results.columns = pd.MultiIndex.from_product(
    [DATASETS, sim_result_cols])

# task_metric_result_cols = ['$Acc_{LC}$', '$Forget_{LC}$']
# if nmc:
#     task_metric_result_cols += ['$Acc_{NMC}$',  '$Forget_{NMC}$']
# task_metric_results = pd.DataFrame(
#     task_metric_results, 
#     columns=task_metric_result_cols,
#     index=DATASET_NAMES)

# sim_results.to_csv('experiments/results_frozen_features.sim', float_format='%.3f')
sim_results.to_latex(
    'figures/multihead-sim-metrics.tex', 
    float_format='%.3f', 
    escape=False, 
    multicolumn=True, 
    multicolumn_format='c', 
    position='h')

# # task_metric_results.to_csv('experiments/results_frozen_features.task', float_format='%.2f')
# task_metric_results.to_latex(
#     'figures/multihead-task-metrics.tex', float_format='%.2f', escape=False)
# print(sim_results)
# print(task_metric_results)
