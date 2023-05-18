from functools import reduce
from typing import List, Union

import numpy as np


def common_n_largest(dfs, n_largest=50):
    """Return common top performing networks of dataset pairs."""
    largest_dfs = {k: df.nlargest(n_largest, 'val_accs') for k, df in dfs.items()}

    common_nets = np.zeros((len(largest_dfs), len(largest_dfs)))
    indices = {k: i for i, k in enumerate(largest_dfs.keys())}

    # get common architectures
    for k1, ldf_1 in largest_dfs.items():
        for k2, ldf_2 in largest_dfs.items():
            i, j = indices[k1], indices[k2]
            if i > j:
                common_nets[i, j] = common_nets[j, i]
                continue

            common_nets[i, j] = len(ldf_1) if k1 == k2 else len(set(ldf_1['net']).intersection(set(ldf_2['net'])))

    return {i: k for k, i in indices.items()}, common_nets


def above_quantile(df, key, quantile=0.9):
    quantile = df[key].quantile(quantile)
    return df[df[key] > quantile]


def filter_by_zc(df, filter_zc: List[str], quantiles: Union[float, List[float]], mode='u'):
    if not isinstance(quantiles, float):
        assert len(filter_zc) == len(quantiles)

    # return None if some proxy is not computed for this df
    for fz in filter_zc:
        if fz not in df.columns:
            return None

    def above_q(data, i):
        q = quantiles[i] if isinstance(quantiles, list) else quantiles
        return data[data[filter_zc[i]] > q].index

    # only one filter proxy
    if len(filter_zc) == 1:
        return above_q(df, 0)

    # stocking of filter proxies
    if mode == 's' or mode == 'stack':
        top_df = df
        # take top q using the next proxy
        for i, _ in enumerate(filter_zc):
            top_df = above_q(top_df, i)
        return top_df.index

    # union or intersection of proxies
    top_nets = [above_q(df, i) for i, _ in enumerate(filter_zc)]
    if mode == 'u' or mode == 'union':
        index = reduce(lambda ix1, ix2: ix1.union(ix2), top_nets)
    elif mode == 'i' or mode == 'intersection':
        index = reduce(lambda ix1, ix2: ix1.intersection(ix2), top_nets)
    else:
        raise ValueError(f"Invalid mode: {mode}, possible u (union), i (intersection), s (stack).")

    return index
