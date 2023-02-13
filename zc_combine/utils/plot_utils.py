from functools import partial

import matplotlib.pyplot as plt

import seaborn as sns
import numpy as np
from scipy import stats

from zc_combine.ensemble.eval import get_stats_zc


def do_subplots(n_total, columns=3, **kwargs):
    fig, axs = plt.subplots(int(np.ceil(n_total / columns)), columns, **kwargs)
    axs = axs.flatten()

    for i in range(n_total, len(axs)):
        fig.delaxes(axs[i])

    return fig, axs.flatten()


def plot_quantile_line(df, key, ax, vmin, vmax, quantile=0.9, horizontal=True, color='r'):
    q = df[key].quantile(quantile)

    func = ax.hlines if horizontal else ax.vlines
    func(q, vmin, vmax, color=color)

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


def _apply_func(func, dfs):
    return [partial(func, task, dfs) for task, dfs in dfs.items()]


def plot_accuracy_histogram(dfs, figsize=(12, 9), title=None, bins=50, subplots_adjust=0.9):
    if title is None:
        title = "Histogram of validation accuracies per task."

    def plot_func(task, df, ax):
        ax.set_title(task)
        sns.histplot(df['val_accs'], ax=ax, bins=bins)

    return _plot_body(_apply_func(plot_func, dfs), figsize, title, subplots_adjust=subplots_adjust)


def plot_common_networks(common_nets, ind_keys, figsize=None, title=None, n_largest=None):
    assert n_largest is not None or title is not None
    title = title if title is not None else f"Common nets between top {n_largest} nets of searchspaces."

    fig = plt.figure(figsize=figsize)
    fig.suptitle(title)
    names = [ind_keys[i] for i in range(len(common_nets))]

    sns.heatmap(common_nets, annot=True, xticklabels=names, yticklabels=names)
    return fig


def plot_top_quantile_zc(dfs_stats, zc, title, zc_quantile=0.9, **kwargs):
    return plot_networks_zc(dfs_stats, zc, title, all_networks=False, top_line=False, zc_quantile=zc_quantile, **kwargs)


def plot_networks_zc(dfs_stats, zc, title, all_networks=True, top_networks=True, top_line=False, zc_quantile=0.9,
                     x_key='val_accs', figsize=(12, 9), subplots_adjust=None, legend_loc='upper left', legend=None,
                     drop_outliers=True, zscore=3.5, margin=0.05, key='tau'):
    if key == 'tau':
        _tau_str = "$\\tau =$"
    else:
        _tau_str = f"{key} $=$"
    assert all_networks or top_networks

    def plot_func(task, df_stats, ax):
        ax_title = f"{task}: "
        titles = []

        df = df_stats['df']

        if all_networks:
            all_nets = df_stats['all']
            all_nets, tau = df.loc[all_nets['index']], all_nets[key]
            titles.append(f"{_tau_str} {tau}")
            sns.scatterplot(data=all_nets, x=x_key, y=zc, ax=ax)

        if top_networks:
            # plot top n %
            top_nets = df_stats['top_quantile']
            top_nets, tau = df.loc[top_nets['index']], top_nets[key]
            titles.append(f"top nets {_tau_str} {tau}")
            sns.scatterplot(data=top_nets, x=x_key, y=zc, ax=ax)

            # plot top k networks
            k_nets = df.loc[df_stats['top_k']['index']]
            sns.scatterplot(data=k_nets, x=x_key, y=zc, ax=ax)

        if top_line:
            val_df = df.loc[df_stats['all']['index']] if all_networks else df.loc[df_stats['top_quantile']['index']]
            vmin, vmax = val_df[x_key].min(), val_df[x_key].max()
            plot_quantile_line(df, zc, ax, vmin, vmax, quantile=zc_quantile)

        if drop_outliers:
            def _drop(what):
                df = all_nets if all_networks else top_nets
                df_noout = df[(np.abs(stats.zscore(df[what])) < zscore)]
                xmin, xmax = df_noout[what].min(), df_noout[what].max()
                m = margin * (xmax - xmin)
                return xmin - m, xmax + m

            xmin, xmax = _drop(x_key)
            ax.set_xlim(xmin, xmax)
            ymin, ymax = _drop(zc)
            ax.set_ylim(ymin, ymax)

        ax.set_title(ax_title + ', '.join(titles))

    palette = sns.color_palette()
    if not all_networks:
        sns.set_palette(palette=palette[1:])

    plot_funcs = _apply_func(plot_func, dfs_stats)
    fig, axs = _plot_body(plot_funcs, figsize, title, subplots_adjust=subplots_adjust)

    if legend is not None:
        fig.legend(loc=legend_loc, labels=legend)

    sns.set_palette(palette=palette)
    return fig, axs
