import networkx as nx

from naslib.search_spaces.nasbench201.conversions import convert_str_to_op_indices
from zc_combine.fixes.operations import parse_ops_nb201, get_ops_edges_nb201, get_ops_edges_tnb101


def to_graph(edges):
    G = nx.DiGraph()
    for e_from, e_to in edges:
        G.add_edge(e_from, e_to)
    return G


def remap_values(c, val1, val2):
    if c != val1 and c != val2:
        return c
    return val1 if c == val2 else val2


def keep_only_isomorpic_nb201(data, meta, zero_is_1=True):
    nb201_unique = [v['nb201-string'] for k, v in meta['ids'].items() if k == v['isomorph']]
    unique_nets = {str(convert_str_to_op_indices(nu)) for nu in nb201_unique}
    if not zero_is_1:
        unique_nets = {[remap_values(c, '0', '1') for c in n] for n in unique_nets}

    return data[data['net'].isin(unique_nets)].copy()


def _nb201_like_to_graph(net, ops, edge_map):
    ops = {i: op for i, op in enumerate(ops)}
    edges = {k: net[i] for k, i in edge_map.items()}
    return ops, edges


def nb201_to_graph(net_str, net_key='net'):
    net = parse_ops_nb201(net_str, net_key=net_key)
    ops, edges = get_ops_edges_nb201()
    return _nb201_like_to_graph(net, ops, edges)


def tnb101_to_graph(net_str, net_key='net'):
    net = parse_ops_nb201(net_str, net_key=net_key)
    ops, edges = get_ops_edges_tnb101()
    return _nb201_like_to_graph(net, ops, edges)
