import math

import numpy as np
import pandas as pd


def filter_by_range(df, zc, min, max):
    return df[(df[zc] >= min) & (df[zc] <= max)]


def parse_ops_nb201(df):
    ops = df['net'].str.strip('()').str.split(', ').to_list()
    return [[int(i) for i in op] for op in ops]


def parse_ops_nb301(df):
    ops = df['net'].to_list()
    ops_tuples = [eval(op) for op in ops]

    def parse_cell(c):
        return [int(op[1]) for op in c]

    return [(parse_cell(op[0]), parse_cell(op[1])) for op in ops_tuples]


def parse_ops_nb101(df, return_edges=True):
    ops = df['net'].str.strip('()').str.split(', ').to_list()

    def parse_cell(c):
        n_nodes = int(math.sqrt(len(c)))
        index = n_nodes * n_nodes
        op, edges = c[index:], c[:index]
        assert len(op) == n_nodes
        op = [int(o) for o in op]
        return (op, [int(e) for e in edges]) if return_edges else op

    vals = [parse_cell(op) for op in ops]
    if not return_edges:
        return vals
    return [o[0] for o in vals], [o[1] for o in vals]


def get_nb301_cell(ops, i=0, both=False):
    if both:
        return [o[0].extend(o[1]) for o in ops]

    return [o[i] for o in ops]


def count_ops(ops, val, index=None):
    res = [o.count(val) for o in ops]
    return pd.Series(data=res, index=index) if index is not None else np.array(res)


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
