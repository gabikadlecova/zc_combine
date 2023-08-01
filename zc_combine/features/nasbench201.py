import networkx as nx
import numpy as np

from zc_combine.features.conversions import to_graph


def _to_names(op_dict, ops):
    return {ops[o]: v for o, v in op_dict.items()}


def count_ops(net, to_names=False):
    ops, edges = net
    op_counts = {i: 0 for i in ops.keys()}
    for val in edges.values():
        op_counts[val] += 1

    return _to_names(op_counts, ops) if to_names else op_counts


def op_on_pos(net):
    ops, edges = net
    op_positions = {(o, k): 0 for o in ops.keys() for k in edges.keys()}
    for k, val in edges.items():
        op_positions[(val, k)] = 1

    return op_positions


def min_path_len(net, banned, start=1, end=4):
    _, edges = net
    active_edges = {e for e, v in edges.items() if v not in banned}

    G = to_graph(active_edges)
    try:
        return nx.shortest_path_length(G, source=start, target=end)
    except (nx.NodeNotFound, nx.NetworkXNoPath):
        return 5


def max_num_on_path(net, allowed, start=1, end=4):
    _, edges = net

    def compute_weight(start, end, _):
        return -1 if edges[(start, end)] in allowed else 1

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


def node_degree(net, allowed, start=1, end=4):
    _, edges = net
    G = to_graph(edges.keys())
    in_edges = {k: [e for e in G.edges if e[0] == k and edges[e] in allowed] for k in range(start, end + 1)}
    out_edges = {k: [e for e in G.edges if e[1] == k and edges[e] in allowed] for k in range(start, end + 1)}

    get_avg = lambda x: np.mean([len(v) for v in x.values()])

    return {'in_degree': len(in_edges[start]), 'out_degree': len(out_edges[end]), 'avg_in': get_avg(in_edges),
            'avg_out': get_avg(out_edges)}


feature_func_dict = {
    'op_count': count_ops,
    'op_on_position': op_on_pos,
    'min_path_len': min_path_len,
    'max_op_on_path': max_num_on_path,
    'node_degree': node_degree
}
