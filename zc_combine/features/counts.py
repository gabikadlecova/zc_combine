import networkx as nx

from zc_combine.features.conversions import to_graph


def _to_names(op_dict, ops):
    return {ops[o]: v for o, v in op_dict.items()}


def count_ops(ops, edges, to_names=False):
    op_counts = {i: 0 for i in ops.keys()}
    for val in edges.values():
        op_counts[val] += 1

    return _to_names(op_counts, ops) if to_names else op_counts


def op_on_pos(ops, edges):
    op_positions = {(o, k): 0 for o in ops.keys() for k in edges.keys()}
    for k, val in edges.items():
        op_positions[(val, k)] = 1

    return op_positions


def min_path_len(edges, banned, start=1, end=4):
    active_edges = {e for e, v in edges.items() if v not in banned}

    G = to_graph(active_edges)
    try:
        return nx.shortest_path_length(G, source=start, target=end)
    except (nx.NodeNotFound, nx.NetworkXNoPath):
        return 5


def max_num_on_path(edges, allowed, start=1, end=4):
    def compute_weight(start, end, _):
        return 0 if edges[(start, end)] in allowed else 1

    G = to_graph(edges.keys())
    try:
        path = nx.shortest_path(G, source=start, target=end, weight=compute_weight)
    except (nx.NodeNotFound, nx.NetworkXNoPath):
        return 0

    n_on_path = 0
    for i, val in enumerate(path):
        if i == len(path) - 1:
            break
        n_on_path += (1 - compute_weight(val, path[i + 1], None))

    return n_on_path


def count_ingoing(edges, banned):
    pass
