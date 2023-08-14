import math
from itertools import chain, combinations

import numpy as np
import pandas as pd
import seaborn as sns


def get_ops_nb101():
    return ['input', 'output', 'maxpool3x3', 'conv1x1-bn-relu', 'conv3x3-bn-relu']


def get_ops_nb301():
    return ["max_pool_3x3", "avg_pool_3x3", "skip_connect", "sep_conv_3x3", "sep_conv_5x5", "dil_conv_3x3",
            "dil_conv_5x5"]


def get_ops_edges_nb201():
    edge_map = ((1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4))
    edge_map = {val: i for i, val in enumerate(edge_map)}
    ops = ['skip_connect', 'none', 'nor_conv_3x3', 'nor_conv_1x1', 'avg_pool_3x3']
    return ops, edge_map


def get_ops_edges_tnb101():
    edge_map = ((1, 2), (1, 3), (2, 3), (1, 4), (2, 4), (3, 4))
    edge_map = {val: i for i, val in enumerate(edge_map)}
    ops = ['none', 'skip_connect', 'nor_conv_3x3', 'nor_conv_1x1', 'avg_pool_3x3']
    return ops, edge_map


def filter_by_range(df, zc, min, max):
    return df[(df[zc] >= min) & (df[zc] <= max)]


def _parse_str(df, net_key='net'):
    if isinstance(df, pd.Series):
        return [df[net_key].strip('()').split(', ')]

    return df[net_key].str.strip('()').str.split(', ').to_list()


def parse_ops_nb201(df, net_key='net'):
    ops = _parse_str(df, net_key=net_key)
    res = [[int(i) for i in op] for op in ops]
    return res[0] if isinstance(df, pd.Series) else res


def parse_ops_nb301(df):
    ops = df['net'].to_list()
    ops_tuples = [eval(op) for op in ops]

    def parse_cell(c):
        return [int(op[1]) for op in c]

    return [(parse_cell(op[0]), parse_cell(op[1])) for op in ops_tuples]


def parse_ops_nb101(df, net_key='net', return_edges=True):
    ops = _parse_str(df, net_key=net_key)

    def parse_cell(c):
        n_nodes = int(math.sqrt(len(c)))
        index = n_nodes * n_nodes
        op, edges = c[index:], c[:index]
        assert len(op) == n_nodes
        op = [int(o) for o in op]
        return (op, [int(e) for e in edges]) if return_edges else op

    vals = [parse_cell(op) for op in ops]
    if len(vals) == 1:
        return vals[0]

    if not return_edges:
        return vals
    return [o[0] for o in vals], [o[1] for o in vals]


def get_nb301_cell(ops, i=0, both=False):
    if both:
        return [(o[0] + o[1]) for o in ops]

    return [o[i] for o in ops]


def count_ops(ops, val, index=None):
    res = [o.count(val) for o in ops]
    return pd.Series(data=res, index=index) if index is not None else np.array(res)


def count_all_ops(df, ops, op_set):
    op_set = list(op_set)
    subsets = chain.from_iterable(combinations(op_set, k) for k in range(1, len(op_set) + 1))

    for subset in subsets:
        count = sum([count_ops(ops, o, index=df.index) for o in subset])
        df[str(subset)] = count


def plot_clouds(df, counts, prox, vmin=0, vmax=7):
    if isinstance(counts, int):
        counts = (counts,)
    if isinstance(counts, tuple):
        counts = df[str(counts)]

    for c in range(vmin, vmax):
        sns.scatterplot(data=df[counts == c], x='val_accs', y=prox)


def filter_by_ops(ops, op_set, all_present=False):
    index = []
    for i in range(len(ops)):
        ops[i] = [int(o) for o in ops[i]]
        op_row = set(ops[i])

        if len(op_row.union(op_set)) > len(op_set):
            continue

        if all_present:
            if len(op_set.difference(op_row)) > 0:
                continue
        index.append(i)

    return index
