from functools import reduce
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


def filter_by_zc(df, filter_zc, quantile=0.9, mode='u'):
    if isinstance(filter_zc, str) or len(filter_zc) == 1:
        filter_zc = filter_zc if isinstance(filter_zc, str) else filter_zc[0]
        return above_quantile(df, filter_zc, quantile).index

    if mode == 's' or mode == 'stack':
        top_df = df
        # take top q using the next proxy
        for i, zc in enumerate(filter_zc):
            q = quantile[i] if isinstance(quantile, list) else quantile
            top_df = above_quantile(top_df, zc, q)
        return top_df.index

    top_nets = [above_quantile(df, zc, quantile).index for zc in filter_zc]
    if mode == 'u' or mode == 'union':
        index = reduce(lambda ix1, ix2: ix1.union(ix2), top_nets)
    elif mode == 'i' or mode == 'intersection':
        index = reduce(lambda ix1, ix2: ix1.intersection(ix2), top_nets)
    else:
        raise ValueError(f"Invalid mode: {mode}, possible u (union), i (intersection), s (stack).")

    return index


def filter_by_zc_task(dfs, filter_zc, quantile=0.9, mode='u'):
    return {task: filter_by_zc(df, filter_zc, quantile=quantile, mode=mode) for task, df in dfs.items()}


def filter_by_index(df, index=None):
    return df if index is None else df[df.index.isin(index)]


def get_above_quantile(df, key, acc_quantile=0.9, filter_index=None):
    top_nets = above_quantile(df, key, quantile=acc_quantile)
    return filter_by_index(top_nets, index=filter_index)


def get_top_k(df, key, top_k=3, filter_index=None):
    best_nets = df.nlargest(top_k, key)
    return filter_by_index(best_nets, index=filter_index)

