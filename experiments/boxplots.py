from utils.eval_utils import get_run_results, process_run_results, load_config

import pandas as pd
import numpy as np

nmc = True
BASE_DIR = 'results/frozen_features'
DATASETS = ['CIFAR_10', 'CIFAR_100', 'CUB200']
# MODELS = ['ResNet', 'DenseNet', 'VGG']
MODELS = ['ResNet', 'DenseNet']

def format_data_for_df(data, metric, model, dataset):
    return [{
        "Model" : model,
        "Dataset" : dataset,
        "Metric" : metric,
        "Value" : d
    } for d in data]


data = pd.DataFrame(columns=["Model", "Dataset", "Metric", "Value"])

for model in MODELS:
    model_sim_results = []
    for dataset in DATASETS:
        results_dir = f'{BASE_DIR}/{model.lower()}/{dataset.lower()}'
        run_config = load_config(results_dir)
        # assert dataset.lower().split('-_') == run_config["dataset"].lower().split('-_')

        lin_accs, lin_fgts, nmc_accs, nmc_fgts, task_sims = get_run_results(
            results_dir, nmc=nmc)
        data = data.append(format_data_for_df(lin_accs, "Linear Acc", model, dataset), ignore_index=True)
        data = data.append(format_data_for_df(lin_fgts, "Linear Fgt", model, dataset), ignore_index=True)
        data = data.append(format_data_for_df(nmc_accs, "NMC Acc", model, dataset), ignore_index=True)
        data = data.append(format_data_for_df(nmc_fgts, "NMC Fgt", model, dataset), ignore_index=True)

print(data['Dataset']=='CIFAR_10')
print(data[(data['Dataset'] == 'CIFAR_10') & (data['Model'] == 'ResNet')])
