import numpy as np
import pandas as pd
import scipy.stats
from sklearn.metrics import ndcg_score
from zc_combine.ensemble.filter import filter_by_zc, above_quantile
from zc_combine.score import FilterProxyScore


def get_tau(accs, scores, round_places=None):
    tau, _ = scipy.stats.kendalltau(accs, scores)
    return tau if round_places is None else np.round(tau, round_places)


def get_corr(accs, scores, round_places=None):
    corr, _ = scipy.stats.spearmanr(accs, scores)
    return corr if round_places is None else np.round(corr, round_places)


def get_ndcg(accs, scores, round_places=None, top_k=None):
    ndcg = ndcg_score(accs.to_numpy()[np.newaxis], scores.to_numpy()[np.newaxis], ignore_ties=True, k=top_k)
    return ndcg if round_places is None else np.round(ndcg, round_places)


def eval_zerocost_score(df, zc, acc_quantile=0.9, x_key='val_accs', top_k=3, round_tau=2, pad_val=0.0):
    res = {}

    def _get_filtered_stats(filt, stats):
        corr_filt = filt[filt[zc] != pad_val]
        tau = get_tau(corr_filt[x_key], corr_filt[zc], round_places=round_tau)
        corr = get_corr(corr_filt[x_key], corr_filt[zc], round_places=round_tau)


        # ndcg using all data or only filtered data
        stat_dict = {'tau': tau, 'corr': corr}
        for name in ['filt_', '']:
            for k in [None, 10, 50]:
                scores = filt if not len(name) else corr_filt
                nk = f"_{k}" if k is not None else ""

                rank_stats = stats['ranking_full'] if not len(name) else stats['ranking_filter']
                ranks_adjusted = len(df) - rank_stats.loc[scores.index]['rank'] + 1
                stat_dict[f'{name}ndcg{nk}'] = get_ndcg(ranks_adjusted, scores[zc], round_places=round_tau, top_k=k)

        return stat_dict

    stats = get_accuracy_ranks(df, zc, pad_val=pad_val, x_key=x_key)

    res['stats'] = stats

    metrics = _get_filtered_stats(df, stats)
    res['all'] = {**metrics, 'index': df[df[zc] != pad_val].index}

    # top n %
    top_nets = above_quantile(df, x_key, quantile=acc_quantile)

    metrics = _get_filtered_stats(top_nets, stats)
    res['top_quantile'] = {**metrics, 'quantile': acc_quantile, 'index': top_nets[top_nets[zc] != pad_val].index}

    # top k networks
    top_nets = df.nlargest(top_k, x_key)
    top_nets = top_nets[top_nets[zc] != pad_val]
    res['top_k'] = {'k': top_k, 'index': top_nets.index}

    res['df'] = df

    return res


def get_accuracy_ranks(df, zc, pad_val=0.0, x_key='val_accs', funcs=None):
    # split to filtered and left out
    stats_df = df[[x_key, zc, 'rank']].copy()
    acc_dropped, acc_filtered = stats_df[stats_df[zc] == pad_val], stats_df[stats_df[zc] != pad_val]
    res = {'ranking_full': stats_df, 'ranking_filter': acc_filtered, 'ranking_drop': acc_dropped}

    # compute ranking stats
    funcs = funcs if funcs is not None else {'median': np.median, 'min': np.min, 'mean': np.mean, 'max': np.max}

    for fname, f in funcs.items():
        res[f"acc_{fname}"], res[f"rank_{fname}"] = f(acc_filtered[x_key]), f(acc_filtered['rank'])
        res[f"acc_{fname}_drop"], res[f"rank_{fname}_drop"] = f(acc_dropped[x_key]), f(acc_dropped['rank'])
        res[f"acc_{fname}_full"], res[f"rank_{fname}_full"] = f(stats_df[x_key]), f(stats_df['rank'])

    return res


def eval_combined_proxies(df, zc_quantile=0.9, key='tau', pad_val=0.0, **kwargs):
    proxies = [c for c in df.columns if c not in ['net', 'val_accs', 'rank']]

    scores = np.zeros((len(proxies), len(proxies)))
    inds = {p: i for i, p in enumerate(proxies)}

    res_df = df[['val_accs', 'rank']].copy()

    for filter_zc in proxies:
        for rank_zc in proxies:
            scorer = FilterProxyScore(filter_zc, rank_zc, quantile=zc_quantile, pad_val=pad_val)
            scorer.fit(df)
            res_df['score'] = scorer.predict(df)

            stats = eval_zerocost_score(res_df, 'score', pad_val=pad_val, **kwargs)

            val = stats['all'][key]
            scores[inds[filter_zc], inds[rank_zc]] = val

    return {p: i for i, p in inds.items()}, scores


def get_stats_ranks(dfs):
    ranks = {t: {k: v for k, v in d['stats'].items() if 'ranking' in k} for t, d in dfs.items()}
    stats = {t: {k: v for k, v in d['stats'].items() if k != 'index' and 'ranking' not in k} for t, d in dfs.items()}

    # also include tau and correlation to the df
    for t, d in dfs.items():
        for key, key_name in [('all', ''), ('top_quantile', 'top_')]:
            for k in d[key].keys():
                if not isinstance(d[key][k], float):
                    continue
                stats[t][f'{key_name}{k}'] = d[key][k]

    return pd.DataFrame(stats), ranks
