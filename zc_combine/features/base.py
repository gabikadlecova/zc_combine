import networkx as nx

from zc_combine.features.conversions import to_graph


def _to_names(op_dict, ops):
    return {ops[o]: v for o, v in op_dict.items()}


def count_ops(net, to_names=False):
    ops, edges = net
    op_counts = {i: 0 for i in ops.keys()}
    for val in edges.values():
        op_counts[val] += 1

    return _to_names(op_counts, ops) if to_names else op_counts


def get_in_out_edges(edges, allowed, start=1, end=4):
    G = to_graph(edges.keys())
    in_edges = {k: [e for e in G.edges if e[0] == k and edges[e] in allowed] for k in range(start, end + 1)}
    out_edges = {k: [e for e in G.edges if e[1] == k and edges[e] in allowed] for k in range(start, end + 1)}
    return in_edges, out_edges


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
