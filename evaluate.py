from pathlib import Path
from scipy.stats import pearsonr
import pandas as pd
import numpy as np
import torch
from torch.nn.functional import cosine_similarity
from typing import List
import math
from utils.dataset_utils import DATASETS
from utils.task2vec_utils import cos_similarity
import json

def rounddown(x):
    return int(math.floor(x / 10.0)) * 10

def parse_acc_forgetting(results) -> List:
    mean_accs = []
    mean_fgts = []
    num_tasks = results.shape[0]
    for i in range(1, num_tasks):
        mean_acc = results[i][:i + 1].mean()
        mean_fgt = np.mean([results[j][j] - results[i][j] for j in range(i)])
        mean_accs.append(mean_acc)
        mean_fgts.append(mean_fgt)
    return mean_accs, mean_fgts


def parse_task_vecs(embeddings, idx, type='linear') -> List:
    norm_sims = []
    sims = []
    norm_old = []
    norm_new = []
    for i in range(len(embeddings)):
        if type == 'linear':
            old_emb = embeddings[i]['linear_old_vec']
            new_emb = embeddings[i]['linear_new_vec']
        else:
            old_emb = embeddings[i]['prototype_old_vec']
            new_emb = embeddings[i]['prototype_new_vec']
        old_emb = old_emb[idx[0]:idx[1]]
        new_emb = new_emb[idx[0]:idx[1]]

        sims.append(cos_similarity(old_emb, new_emb, norm=False))
        norm_sims.append(cos_similarity(old_emb, new_emb, norm=True))
        norm_old.append(torch.linalg.norm(old_emb).cpu())
        norm_new.append(torch.linalg.norm(new_emb).cpu())
    return np.array(sims), np.array(norm_sims), np.array(norm_old), np.array(norm_new)

def load_results(run_dir: Path, load_embeddings=True):
    results = pd.read_csv(run_dir / 'results.csv')
    results = results.to_numpy()
    results = results[:,1:]
    if run_dir.name == 'nmc':
        run_dir = str(run_dir.absolute())
        run_dir = run_dir.replace("nmc", "linear")
    embeddings = None
    if load_embeddings:
        embeddings = torch.load(f'{run_dir}/embeddings.pt')
    return results, embeddings

# MODEL_FE_PARAMS = {
#     'resnet': rounddown(23508032),
#     'densenet': rounddown(6953856)
# }

def get_corr_layers(model):
    path = str(Path('config') / model) + '.json'
    with open(path, 'r') as f:
        model_layer_to_idx = json.load(f)
    model_layer_to_idx = list(model_layer_to_idx.items())
    model_layer_to_idx.sort(key=lambda x: x[1]) ## Sort by idx
    assert 'classifier' in model_layer_to_idx[-1][0].lower()
    all_features_idx = [model_layer_to_idx[0][1], model_layer_to_idx[-1][1]]
    final_layer_idx = [model_layer_to_idx[-2][1], model_layer_to_idx[-1][1]]
    return [
        ('all', all_features_idx),
        ('final', final_layer_idx)
    ]

def evaluate_results(args, results, embeddings=None):
    accs, fgts = parse_acc_forgetting(results)
    summary = dict(all_cl_accs=accs, all_fgts=fgts)
    if embeddings:
        for (layer, idx) in get_corr_layers(args.model):
            sims, norm_sims, norm_old, norm_new = parse_task_vecs(embeddings, idx,  'linear')
            summary[f'{layer}_sims'] = sims
            summary[f'{layer}_norm_sims'] = norm_sims
            summary[f'{layer}_norm_old'] = norm_old
            summary[f'{layer}_norm_new'] = norm_new
    return summary


def get_corr_stats(sim_type, embeddings, model, dataset, seed, head_type, accs, fgts, transfer=None):
    corr_stats = []
    for (layer, idx) in get_corr_layers(model):
        sims, norm_old, norm_new, running_sims = parse_task_vecs(embeddings, sim_type, idx)
        corr_acc, corr_acc_p = pearsonr(accs, running_sims)
        corr_fgt, corr_fgt_p = pearsonr(fgts, running_sims)
        stats = {
            'corr_acc': corr_acc,
            'corr_acc_p': corr_acc_p,
            'corr_fgt': corr_fgt,
            'corr_fgt_p': corr_fgt_p,
            'head_type': head_type,
            'layer': layer,
            'dataset': dataset,
            'seed': seed,
            'model': model,
            'sim_type': sim_type,
            'sims' : sims,
            'norm_old' : norm_old,
            'norm_new' : norm_new,
            'running_sims' : running_sims
        }
        if transfer is not None:
            stats['corr_transfer'] = pearsonr(transfer, running_sims)[0]
        corr_stats.append(stats)
    return corr_stats

def process_results(base_dir, init):
    base_dir = Path(base_dir)
    perf_stats = []
    corr_stats = []
    for dataset in [d for d in list(base_dir.iterdir()) if d.name in DATASETS.keys()]:
        for model in dataset.iterdir():
            for seed in model.iterdir():
                for type in seed.iterdir():
                    results, embeddings = load_results(type.absolute(), load_embeddings=(not init))
                    # fix_embeddings(embeddings, model.name)
                    accs, fgts = parse_acc_forgetting(results)
                    perf_stats.append({
                        'model' : model.name,
                        'seed' : seed.name,
                        'head_type' : type.name,
                        'dataset' : dataset.name,
                        'final_mean_acc' : accs[-1],
                        'final_mean_fgt' : fgts[-1]
                    })
                    if not init:
                        for sim_type in ['linear']:
                            corr_stats.extend(get_corr_stats(
                                sim_type=sim_type,
                                embeddings=embeddings, 
                                model=model.name, 
                                dataset=dataset.name, 
                                seed=seed.name, 
                                head_type=type.name, 
                                accs=accs, 
                                fgts=fgts))
    pd.to_pickle(pd.DataFrame(perf_stats), base_dir / 'results/perf_results.pk')
    if not init:
        pd.to_pickle(pd.DataFrame(corr_stats), base_dir / 'results/corr_results.pk')

for setting in Path('results').iterdir():
    process_results(setting, init=('init' in setting.name.lower()))

# if __name__ == '__main__':
#     import argparse
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--dir', required=True)
#     parser.add_argument('--init', action='store_true')
#     args = parser.parse_args()
#     process_results(args.dir, init=args.init)