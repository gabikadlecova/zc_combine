import numpy as np
import scipy.stats
from zc_combine.ensemble.filter import filter_by_index, get_above_quantile, get_top_k, filter_by_zc


def get_tau(accs, scores, round_places=None):
    tau, _ = scipy.stats.kendalltau(accs, scores)
    return tau if round_places is None else np.round(tau, round_places)


def get_corr(accs, scores, round_places=None):
    corr, _ = scipy.stats.spearmanr(accs, scores)
    return corr if round_places is None else np.round(corr, round_places)


def get_accuracy_ranks(df, filter_index, x_key='val_accs', funcs=None):
    # split to filtered and left out
    stats_df = df[[x_key]].copy()
    stats_df['rank'] = stats_df[x_key].rank(ascending=False)
    acc_dropped, acc_filtered = stats_df.drop(index=filter_index), stats_df.loc[filter_index]
    res = {'ranking_full': stats_df, 'ranking_filter': acc_filtered, 'ranking_drop': acc_dropped}

    # compute ranking stats
    funcs = funcs if funcs is not None else {'median': np.median, 'min': np.min, 'mean': np.mean, 'max': np.max}

    for fname, f in funcs.items():
        res[f"acc_{fname}"], res[f"rank_{fname}"] = f(acc_filtered[x_key]), f(acc_filtered['rank'])
        res[f"acc_{fname}_drop"], res[f"rank_{fname}_drop"] = f(acc_dropped[x_key]), f(acc_dropped['rank'])
        res[f"acc_{fname}_full"], res[f"rank_{fname}_full"] = f(stats_df[x_key]), f(stats_df['rank'])

    return res


def get_stats_zc(df, zc, acc_quantile=0.9, top_k=3, x_key='val_accs', round_tau=2, filter_index=None, include_df=True):
    res = {}

    def _get_filtered_stats(filt):
        tau = get_tau(filt[x_key], filt[zc], round_places=round_tau)
        corr = get_corr(filt[x_key], filt[zc], round_places=round_tau)
        return tau, corr

    filtered_df = filter_by_index(df, index=filter_index)
    tau, corr = _get_filtered_stats(filtered_df)
    stats = get_accuracy_ranks(df, filtered_df.index, x_key=x_key)

    res['stats'] = stats
    res['all'] = {'tau': tau, 'corr': corr, 'index': filtered_df.index}

    # top n %
    top_nets = get_above_quantile(df, x_key, acc_quantile=acc_quantile, filter_index=filter_index)
    tau, corr = _get_filtered_stats(top_nets)
    res['top_quantile'] = {'tau': tau, 'corr': corr, 'quantile': acc_quantile, 'index': top_nets.index}

    # top k networks
    top_nets = get_top_k(df, x_key, top_k=top_k, filter_index=filter_index)
    res['top_k'] = {'k': top_k, 'index': top_nets.index}

    if include_df:
        res['df'] = df

    return res


def eval_combined_proxies(df, zc_quantile=0.9, key='tau', **kwargs):
    proxies = [c for c in df.columns if c not in ['net', 'val_accs']]

    scores = np.zeros((len(proxies), len(proxies)))
    inds = {p: i for i, p in enumerate(proxies)}

    for filter_zc in proxies:
        for rank_zc in proxies:
            df_filter = filter_by_zc(df, filter_zc, quantile=zc_quantile)
            df_filtered = get_stats_zc(df, rank_zc, filter_index=df_filter, **kwargs)

            tau = df_filtered['all'][key]
            scores[inds[filter_zc], inds[rank_zc]] = tau

    return {p: i for i, p in inds.items()}, scores
