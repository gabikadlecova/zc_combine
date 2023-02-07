import numpy as np
import scipy.stats
from zc_combine.ensemble.filter import filter_by_index, get_above_quantile, get_top_k, filter_by_zc_task


def get_tau(accs, scores, round_places=None):
    tau, _ = scipy.stats.kendalltau(accs, scores)
    return tau if round_places is None else np.round(tau, round_places)


def get_corr(accs, scores, round_places=None):
    corr, _ = scipy.stats.spearmanr(accs, scores)
    return corr if round_places is None else np.round(corr, round_places)


def get_stats_zc(df, zc, acc_quantile=0.9, top_k=3, x_key='val_accs', round_tau=2, filter_index=None, include_df=True):
    res = {}

    filtered_df = filter_by_index(df, index=filter_index)
    tau = get_tau(filtered_df[x_key], filtered_df[zc], round_places=round_tau)
    corr = get_corr(filtered_df[x_key], filtered_df[zc], round_places=round_tau)
    res['all'] = {'tau': tau, 'corr': corr, 'index': filtered_df.index}

    # top n %
    top_nets = get_above_quantile(df, x_key, acc_quantile=acc_quantile, filter_index=filter_index)
    tau = get_tau(top_nets[x_key], top_nets[zc], round_places=round_tau)
    corr = get_corr(top_nets[x_key], top_nets[zc], round_places=round_tau)
    res['top_quantile'] = {'tau': tau, 'corr': corr, 'quantile': acc_quantile, 'index': top_nets.index}

    # top k networks
    top_nets = get_top_k(df, x_key, top_k=top_k, filter_index=filter_index)
    res['top_k'] = {'k': top_k, 'index': top_nets.index}

    if include_df:
        res['df'] = df

    return res


def eval_zc(dfs, zc, filter_index=None, **kwargs):
    def get_idx(task):
        return None if filter_index is None else filter_index[task]

    return {task: get_stats_zc(df, zc, filter_index=get_idx(task), **kwargs) for task, df in dfs.items()}


def eval_combined_proxies(dfs, zc_list, zc_quantile=0.9, key='tau', **kwargs):
    n_proxies = len(zc_list)
    scores = {task: np.zeros((n_proxies, n_proxies)) for task in dfs.keys()}
    inds = {p: i for i, p in enumerate(zc_list)}

    for filter_zc in zc_list:
        for rank_zc in zc_list:
            dfs_filter = filter_by_zc_task(dfs, filter_zc, quantile=zc_quantile)
            dfs_filtered = eval_zc(dfs, rank_zc, filter_index=dfs_filter, **kwargs)

            for task, df in dfs_filtered.items():
                tau = df['all'][key]
                scores[task][inds[filter_zc], inds[rank_zc]] = tau

    return {p: i for i, p in inds.items()}, scores
