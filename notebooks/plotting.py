import re
import pandas as pd
import wandb
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import numpy as np
from pathlib import Path
sns.set()

api = wandb.Api()
METRICS = ['entropy_ratio', 'entropy_diff', 'max_prob_diff',
                     'max_prob_ratio', 'max_logit_diff', 'max_logit_ratio', 'kl_div', 'all_sims']

def get_iid_accs(r):
    accs = []
    if 'output.log' in [f.name for f in r.files()]:
        r.file('output.log').download(replace=True)
        with open('output.log', 'r') as f:
            for line in f:
                if re.search('Task \d IID Test Accuracy: (.*)$\n', line):
                    iid = float(
                        re.match('Task \d IID Test Accuracy: (.*)$\n', line).groups()[0])
                    accs.append(iid)
    return accs


def load_sweeps(sweeps, parse_iid=False):
    dfs = []
    for s in sweeps:
        if Path(f'saved_dfs/{s}').exists():
            dfs.append(pd.read_pickle(Path(f'saved_dfs/{s}')))
        else:
            sweep = api.sweep(f"clip_cl/CL-Similarity/{s}")
            results = []
            for run in sweep.runs:
                summary = {k: v for k, v in run.summary._json_dict.items()
                           if not k.startswith('_')}
                config = {k: v for k, v in run.config.items()
                          if not k.startswith('_')}
                name = {'name': run.name}
                res = summary | config | name
                if parse_iid:
                    res['iid_accs'] = get_iid_accs(run)
                results.append(res)
            my_df = pd.DataFrame(results)
            my_df.to_pickle(Path(f'saved_dfs/{s}'))
            dfs.append(my_df)
    df = pd.concat(dfs)
    df.loc[df.replay_size_per_class == -1,
           'replay_size_per_class'] = 'All samples'
    for m in METRICS:
        df = df.rename(columns={m: f'metric_{m}'})
    return df

def combine_transfer_df(df, transfer_df):
    mean_transfer = []
    all_transfers = []
    for i in range(df.shape[0]):
        df_row = df.iloc[i]
        df_init_row = transfer_df.loc[
            (transfer_df.dataset == df_row.dataset) &
            (transfer_df.n_classes_per_task == df_row.n_classes_per_task) &
            (transfer_df.model == df_row.model) &
            (transfer_df.seed == df_row.seed)
        ]
        cl_iid_accs = df_row.iid_accs
        init_iid_accs = df_init_row.iid_accs.values[0]

        if len(cl_iid_accs) != len(init_iid_accs):
            mean_transfer.append(0)
            all_transfers.append(np.zeros(4))
        else:
            mean_transfer.append(
                (df_row['mean_iid_acc'] - df_init_row['mean_iid_acc']).values[0])
            all_t = np.array(cl_iid_accs) - np.array(init_iid_accs)
            all_transfers.append(all_t[1:]/100)
    df['mean_transfer'] = np.array(mean_transfer)
    df['all_transfers'] = all_transfers
    return df

def load_results_df(metric_sweep_names, transfer_sweep_names):
    df = load_sweeps(metric_sweep_names)
    transfer_df = load_sweeps(transfer_sweep_names)
    df = combine_transfer_df(df, transfer_df)
    df['task_idx'] = [np.arange(4) + 1 for i in range(df.shape[0])]

    metrics = get_metrics(df)

    blowup_cols = ['all_cl_accs', 'all_fgts',
                   'task_idx', 'all_transfers'] + metrics
    metrics_df = df.explode(blowup_cols, ignore_index=True)
    for col in blowup_cols:
        metrics_df[col] = np.float64(metrics_df[col])
    return df, metrics_df
    

def get_metrics(df):
    return [c for c in df.columns if c.startswith('metric_')]


def create_corr_df(df, sim_metrics, agg_cols=['model', 'dataset', 'n_classes_per_task'], idx=False):
    df = df.copy()
    if idx:
        agg_cols.append('task_idx')

    def dict_product(dicts):
        return list(dict(zip(dicts, x)) for x in itertools.product(*dicts.values()))

    corr_results = []
    combinations = dict_product({x: list(df[x].unique()) for x in agg_cols})
    for c in combinations:
        tmp_df = df.loc[(df[list(c)] ==
                         pd.Series(c)).all(axis=1)]
        for i in ['all_cl_accs', 'all_fgts', 'all_transfers']:
            for j in sim_metrics:
                c[f'{i}_{j}'] = tmp_df[i].corr(tmp_df[j])
        corr_results.append(c)
    return pd.DataFrame(corr_results)


def create_corr_pivot_table(df, values='all_cl_accs_all_sims', index=['model', 'dataset'], columns=['n_classes_per_task'], savefig=None):
    df = df.pivot_table(values=values, index=index, columns=columns,
                        aggfunc=np.sum).rename_axis(columns=None)
    if savefig:
        t = df.style.format(precision=2)
        t.to_latex(buf=savefig, hrules=True)
    return df



## Make Plots

def plot_perf(df, x, hue, figsize=(20, 10), savefig=None):
    fig, ax = plt.subplots(1, 4, figsize=figsize)
    for i, y in enumerate(['mean_iid_acc', 'cl_acc', 'mean_fgt', 'mean_transfer']):
        sns.barplot(data=df, x=x, y=y, hue=hue, ci='sd', ax=ax[i])
        ax[i].set_ylim(0, 1)
        ax[i].set_ylabel(['IID Acc', 'CL Acc', 'Forgetting', 'Transfer'][i])
    fig.tight_layout()
    if savefig:
        fig.savefig(savefig)


def plot_single_metric(df, x, y, savefig=None, **plot_args):
    fig, ax = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    palettes = ['Accent_r', 'inferno', 'Dark2']
    for i, hue in enumerate(['task_idx', 'model', 'n_classes_per_task']):
        plot = sns.scatterplot(
            data=df, ax=ax[i], x=x, y=y, hue=hue, palette=palettes[i])
        ax[i].legend(loc='lower left', title=hue)
        ax[i].set_xlim(0, 1)
        if plot_args:
            plot.set(**plot_args)
    if savefig:
        fig.savefig(savefig)


def plot_single_metric_split(df, x, y, figsize=(12, 15), savefig=None, model=None, n_classes_per_task=[2, 4, 10, 20], ylim=None, xlim=None):
    fig = plt.figure(constrained_layout=True, figsize=(
        len(n_classes_per_task) * 5, 20))
    subfigs = fig.subfigures(4, 1, wspace=0.07)

    if model:
        df = df.loc[df['model'] == model]

    def make_plot(df, ax, hue):
        hue_order = sorted(df[hue].unique())
        sns.scatterplot(data=df, ax=ax, x=x,
                        y=y, hue=hue, hue_order=hue_order)

    for i, task_idx in enumerate([1, 2, 3, 4]):
        # subfigs[i].suptitle(d[1], fontweight="bold")
        axes = subfigs[i].subplots(1, len(n_classes_per_task))
        subfigs[i].supylabel(f'After task: {task_idx + 1}')
        for j, n_class_per_task in enumerate(n_classes_per_task):
            if len(n_classes_per_task) == 1:
                ax = axes
            else:
                ax = axes[j]
            ax_df = df.loc[
                (df['n_classes_per_task'] == n_class_per_task) &
                (df['task_idx'] == task_idx)]
            hue = 'dataset'
            make_plot(ax_df, ax, hue)
            if ylim:
                ax.set_ylim(*ylim)
            if xlim:
                ax.set_xlim(*xlim)
            ax.set_xlabel(None)
            ax.set_ylabel(None)
            ax.set_title(
                f'N Classes per task: {n_class_per_task}' if i == 0 else None)
            if not model:
                ax.legend(loc='lower right')
    if savefig:
        fig.savefig(savefig)


def corr_bar_plot(df, x='n_classes_per_task', xlabel=None, final_layer=False, savefig=None):
    if final_layer:
        ys = ['all_cl_accs_final_sims',
              'all_fgts_final_sims', 'all_transfers_final_sims']
    else:
        ys = ['all_cl_accs_all_sims',
              'all_fgts_all_sims', 'all_transfers_all_sims']
    fig, ax = plt.subplots(1, 3, figsize=(9, 3), constrained_layout=True)
    sns.barplot(data=df, x=x, y=ys[0], ax=ax[0])
    sns.barplot(data=df, x=x, y=ys[1], ax=ax[1])
    sns.barplot(data=df, x=x, y=ys[2], ax=ax[2])
    ax[0].set_ylabel('CL Acc Corr')
    ax[1].set_ylabel('Forget Corr')
    ax[2].set_ylabel('Transfer Corr')
    for a in ax:
        a.set_ylim(-1, 1)
        if xlabel:
            a.set_xlabel(xlabel)
    if savefig:
        fig.savefig(savefig)

