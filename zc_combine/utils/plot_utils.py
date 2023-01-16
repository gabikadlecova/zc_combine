from functools import partial

import matplotlib.pyplot as plt

import seaborn as sns
import numpy as np

from zc_combine.ensemble.eval import get_stats_zc


def do_subplots(n_total, columns=3, **kwargs):
    fig, axs = plt.subplots(int(np.ceil(n_total / columns)), columns, **kwargs)
    axs = axs.flatten()

    for i in range(n_total, len(axs)):
        fig.delaxes(axs[i])

    return fig, axs.flatten()


def plot_quantile_line(df, key, ax, quantile=0.9, horizontal=True, color='r'):
    q = df[key].quantile(quantile)

    func = ax.hlines if horizontal else ax.vlines
    func(q, df[key].min(), df[key].max(), color=color)

    return ax


def _plot_body(funcs, figsize, title, subplots_adjust=None):
    fig, axs = do_subplots(len(funcs), figsize=figsize)

    for func, ax in zip(funcs, axs):
        func(ax)

    fig.suptitle(title)
    fig.tight_layout()
    if subplots_adjust is not None:
        plt.subplots_adjust(top=subplots_adjust)

    return fig, axs


#df_stats = get_stats_zc(df, zc, acc_quantile=acc_quantile, top_k=top_k, x_key=x_key, round_tau=round_tau,
#                        filter_index=filter_index)


def plot_networks_zc(dfs, zc, title,  all_networks=True, top_networks=True, top_line=False, zc_quantile=0.9,
                     x_key='val_accs', figsize=(12, 9), subplots_adjust=None, legend_loc='upper left', legend=None):
    _tau_str = "$\\tau =$"

    def plot_func(task, df_stats, ax):
        ax_title = f"{task}: "
        titles = []

        df = df_stats['df']

        if all_networks:
            all_nets = df_stats['all']
            all_nets, tau = df.loc[all_nets['index']], all_nets['tau']
            titles.append(f"{_tau_str} {tau}")
            sns.scatterplot(data=all_nets, x=x_key, y=zc, ax=ax)

        if top_networks:
            # plot top n %
            top_nets = df_stats['top_quantile']
            top_nets, tau = df.loc[top_nets['index']], top_nets['tau']
            titles.append(f"top nets {_tau_str} {tau}")
            sns.scatterplot(data=top_nets, x=x_key, y=zc, ax=ax)

            # plot top k networks
            top_nets = df.loc[df_stats['top_k']['index']]
            sns.scatterplot(data=top_nets, x=x_key, y=zc, ax=ax)

        if top_line:
            plot_quantile_line(df, zc, ax, quantile=zc_quantile)

        ax.set_title(ax_title + ', '.join(titles))

    plot_funcs = [partial(plot_func, task, df_stats) for task, df_stats in dfs.items()]
    fig, axs = _plot_body(plot_funcs, figsize, title, subplots_adjust=subplots_adjust)

    if legend is not None:
        fig.legend(loc=legend_loc, labels=legend)

    return fig, axs
