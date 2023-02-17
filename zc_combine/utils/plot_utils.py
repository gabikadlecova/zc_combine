from functools import partial
from typing import List, Union

import matplotlib.pyplot as plt
import pandas as pd

import seaborn as sns
import numpy as np
from scipy import stats


def plot_accuracy_histogram(dfs, figsize=(12, 9), title=None, bins=50, subplots_adjust=0.9):
    fig, axs = do_subplots(len(dfs), figsize=figsize)

    if title is None:
        title = "Histogram of validation accuracies per task."

    for tdf, ax in zip(dfs.items(), axs):
        task, df = tdf
        ax.set_title(task)
        sns.histplot(df['val_accs'], ax=ax, bins=bins)

    _plot_body(fig, title, subplots_adjust=subplots_adjust)
    return fig, axs


def plot_common_networks(common_nets, ind_keys, figsize=None, title=None, n_largest=None):
    assert n_largest is not None or title is not None
    title = title if title is not None else f"Common nets between top {n_largest} nets of searchspaces."

    fig = plt.figure(figsize=figsize)
    fig.suptitle(title)
    names = [ind_keys[i] for i in range(len(common_nets))]

    sns.heatmap(common_nets, annot=True, xticklabels=names, yticklabels=names)
    return fig


def plot_top_accuracy_zc(dfs_stats, zc, bench_name, zc_quantile=0.9, top_line=True, **kwargs):
    return plot_networks_by_zc(dfs_stats, zc, bench_name, all_networks=False, top_line=top_line,
                               zc_quantile=zc_quantile, **kwargs)


def plot_networks_by_zc(dfs_stats, zc, bench_name, accuracy_q=0.9, net_k=3, all_networks=True, top_line=True,
                        zc_quantile=0.9, **kwargs):
    suff = "" if all_networks else f" (top {100 - int(accuracy_q * 100)} acc only)"
    title = f"{bench_name} - zero cost proxy score ({zc}) by validation accuracy{suff}."

    legend = _get_legend(accuracy_q, net_k)
    if not all_networks:
        legend = legend[1:]
    if top_line:
        legend.append(f"Zc score - {zc_quantile} quantile")

    plot = _plot_dfs_stats(dfs_stats, zc, title, legend=legend, legend_loc='lower right', all_networks=all_networks,
                           top_line=top_line, zc_quantile=zc_quantile, **kwargs)
    return plot


def plot_filtered_by_zc(dfs, filter_zc, rank_zc, bench_name, key='tau', quantile: Union[float, List[float]] = 0.9,
                        accuracy_q=0.9, net_k=3, zscore=3.5, **kwargs):

    qtitle = int(quantile * 100) if isinstance(quantile, float) else [int(q * 100) for q in quantile]
    title = f"{bench_name} - nets over {qtitle}% quantile in {filter_zc}, {rank_zc} by validation accuracy."
    legend = _get_legend(accuracy_q, net_k)

    plot = _plot_dfs_stats(dfs, rank_zc, title, legend=legend, legend_loc='lower right',
                           key=key, zscore=zscore, **kwargs)
    return plot


def plot_filtered_ranks(stats, key='rank', figsize=None):
    assert key == 'rank' or key == 'val_accs'
    stats = {k: stats[k].copy() for k in ['ranking_filter', 'ranking_drop']}

    stats['ranking_filter']['name'] = 'filter'
    stats['ranking_drop']['name'] = 'drop'
    rdata = pd.concat([stats['ranking_filter'], stats['ranking_drop']])

    fig = plt.figure(figsize=figsize)
    fig.suptitle(f"Histograms of filtered and dropped network {'ranks' if key == 'rank' else 'accuracies'}.")
    sns.histplot(data=rdata, x=key, hue='name')
    return fig


def _get_title(key):
    if key == 'tau':
        return "$\\tau =$"
    else:
        return f"{key} $=$"


def plot_zc_per_df(task, df_stats, zc, ax=None, x_key='val_acc', all_networks=True, top_networks=True, metric_key='tau',
                   top_line=False, zc_quantile=0.9, drop_outliers=True, margin=0.05, zscore=3.5):

    assert all_networks or top_networks

    if ax is None:
        ax = plt.subplot()

    ax_title = f"{task}: "
    titles = []
    metric_title = _get_title(metric_key)

    df = df_stats['df']

    if all_networks:
        all_nets = df_stats['all']
        all_nets, metric = df.loc[all_nets['index']], all_nets[metric_key]
        titles.append(f"{metric_title} {metric}")
        sns.scatterplot(data=all_nets, x=x_key, y=zc, ax=ax)

    if top_networks:
        # plot top n %
        top_nets = df_stats['top_quantile']
        top_nets, metric = df.loc[top_nets['index']], top_nets[metric_key]
        titles.append(f"top nets {metric_title} {metric}")
        sns.scatterplot(data=top_nets, x=x_key, y=zc, ax=ax)

        # plot top k networks
        k_nets = df.loc[df_stats['top_k']['index']]
        sns.scatterplot(data=k_nets, x=x_key, y=zc, ax=ax)

    if top_line:
        # plot line indicating zc_quantile
        val_df = df.loc[df_stats['all']['index']] if all_networks else df.loc[df_stats['top_quantile']['index']]
        vmin, vmax = val_df[x_key].min(), val_df[x_key].max()
        plot_quantile_line(df, zc, ax, vmin, vmax, quantile=zc_quantile)

    if drop_outliers:
        def _drop(what):
            df = all_nets if all_networks else top_nets
            df_noout = df[(np.abs(stats.zscore(df[what])) < zscore)]
            xmin, xmax = df_noout[what].min(), df_noout[what].max()
            if np.isnan(xmin) and np.isnan(xmax):
                return -1, 1
            m = margin * (xmax - xmin)
            return xmin - m, xmax + m

        # adjust limits to inliers
        xmin, xmax = _drop(x_key)
        ax.set_xlim(xmin, xmax)
        ymin, ymax = _drop(zc)
        ax.set_ylim(ymin, ymax)

    ax.set_title(ax_title + ', '.join(titles))
    return ax


def _plot_dfs_stats(dfs_stats, zc, title, all_networks=True, top_networks=True, top_line=False, zc_quantile=0.9,
                    x_key='val_accs', figsize=(12, 9), subplots_adjust=0.9, legend_loc='upper left', legend=None,
                    drop_outliers=True, zscore=3.5, margin=0.05, key='tau'):
    palette = sns.color_palette()
    if not all_networks:
        sns.set_palette(palette=palette[1:])

    fig, axs = do_subplots(len(dfs_stats), figsize=figsize)

    for df_item, ax in zip(dfs_stats.items(), axs):
        task, df = df_item
        plot_zc_per_df(task, df, zc, ax=ax, x_key=x_key, all_networks=all_networks, top_networks=top_networks,
                       metric_key=key, top_line=top_line, zc_quantile=zc_quantile, drop_outliers=drop_outliers,
                       margin=margin, zscore=zscore)

    if legend is not None:
        fig.legend(loc=legend_loc, labels=legend)

    _plot_body(fig, title, subplots_adjust=subplots_adjust)
    sns.set_palette(palette=palette)
    return fig, axs


def _get_legend(acc_q, net_k):
    acc_q = int(acc_q * 100)
    acc_q_top = 100 - acc_q
    return [f'lower {acc_q}% nets by accuracy', f'top {acc_q_top}% nets by accuracy', f'best {net_k} networks']


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


def _plot_body(fig, title, subplots_adjust=None):
    fig.suptitle(title)
    fig.tight_layout()
    if subplots_adjust is not None:
        plt.subplots_adjust(top=subplots_adjust)
