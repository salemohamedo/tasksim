import matplotlib.pyplot as plt
import seaborn as sns
# sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.set()


def create_corr_plot(df, y, y_label, out_file=None, figsize=(11.7, 8.27), ds=['cifar-100', 'cub200', 'car196'], **plot_settings):
    if plot_settings == None:
        plot_settings = dict(hue='sim_type', hue_order=[
            'linear', 'prototype'])
    fig, axes = plt.subplots(1, len(ds), figsize=figsize)
    for i, d in enumerate(ds):
        ds_df = df.loc[df['dataset'] == d]
        sns.barplot(x='layer', y=y, data=ds_df, ax=axes[i], **plot_settings)
        axes[i].set_title(d)
        axes[i].set_ylabel(y_label)
        ax = axes[i]
        labels = [item.get_text() for item in ax.get_xticklabels()]
        labels[0] = 'All parameters'
        ax.set_xticklabels(labels)
        ax.legend(loc='upper right', title='Task2Vec Head')
        # ax.set(ylim=(-1, 1))
    fig.tight_layout()

    if out_file:
        fig.savefig(out_file)
    

def plot_perf(df, figsize, out_file=None, ds=['cifar-100', 'cub200', 'car196'], **plot_settings):
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    sns.barplot(data=df, y='final_mean_acc', x='dataset', order=ds, ax=axes[0], **plot_settings)
    axes[0].set_ylabel('Final Mean Accuracy')
    axes[0].legend(loc='upper right')

    sns.barplot(data=df, y='final_mean_fgt', x='dataset',
                order=ds, ax=axes[1], **plot_settings)
    axes[1].set_ylabel('Final Mean Forgetting')
    axes[1].legend(loc='upper right')
    fig.tight_layout()
    if out_file:
        fig.savefig(out_file)


def plot_norms(df, title, y, figsize=(18, 6), **hue_settings):
    models = df['model'].unique()
    settings = sorted(df['setting'].unique())
    fig, axes = plt.subplots(len(models), len(settings), figsize=figsize)
    fig.suptitle(title, fontweight="bold")
    for i, model in enumerate(models):
        for j, setting in enumerate(settings):
            plt_df = df.loc[(df['model'] == model) &
                            (df['setting'] == setting)]
            sns.barplot(data=plt_df, x='idx', y=y,
                        ax=axes[i][j], **hue_settings)
            axes[i][j].set_xlabel('Task')
            axes[i][j].set_ylabel(model if j == 0 else None)
            axes[i][j].set_title(setting if i == 0 else None)
            axes[i][j].legend(loc='upper right')
    fig.tight_layout()

