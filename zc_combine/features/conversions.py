import naslib
import networkx as nx
import numpy as np
from naslib.search_spaces.nasbench101.conversions import convert_tuple_to_spec
from naslib.search_spaces.nasbench201.encodings import encode_adjacency_one_hot_op_indices

from naslib.search_spaces.nasbench301.conversions import convert_compact_to_genotype
from naslib.search_spaces.nasbench201.conversions import convert_str_to_op_indices
from naslib.search_spaces.transbench101.encodings import encode_adjacency_one_hot_transbench_micro_op_indices

from zc_combine.fixes.operations import parse_ops_nb201, get_ops_edges_nb201, get_ops_edges_tnb101, get_ops_nb301, \
    get_ops_nb101, parse_ops_nb101


def to_graph(edges):
    G = nx.DiGraph()
    for e_from, e_to in edges:
        G.add_edge(e_from, e_to)
    return G


def remap_values(c, val1, val2):
    if c != val1 and c != val2:
        return c
    return val1 if c == val2 else val2


def keep_only_isomorpic_nb201(data, meta, zero_is_1=True, net_key='net', copy=True):
    nb201_unique = [v['nb201-string'] for k, v in meta['ids'].items() if k == v['isomorph']]
    unique_nets = {str(convert_str_to_op_indices(nu)) for nu in nb201_unique}
    if not zero_is_1:
        unique_nets = {''.join([remap_values(c, '0', '1') for c in n]) for n in unique_nets}

    res = data[data[net_key].isin(unique_nets)]
    return res.copy() if copy else res


def nb101_to_graph(net, net_key='net'):
    op_map = [*get_ops_nb101()]
    op_map = {o: i for i, o in enumerate(op_map)}
    op_map = {i: o for o, i in op_map.items()}

    ops, edges = parse_ops_nb101(net,net_key=net_key)
    edges = np.array(edges)
    edges = edges.reshape(int(np.sqrt(edges.shape[0])), -1)
    assert edges.shape[0] == edges.shape[1]

    return op_map, ops, nx.from_numpy_array(edges, create_using=nx.DiGraph)


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


def darts_to_graph(genotype):
    """Adapted from darts plotting script."""
    op_map = ['out', *get_ops_nb301()]
    op_map = {o: i for i, o in enumerate(op_map)}
    ops = {i: o for o, i in op_map.items()}
    edges = {}

    assert len(genotype) % 2 == 0
    steps = len(genotype) // 2

    for i in range(steps):
        for k in [2 * i, 2 * i + 1]:
            op, j = genotype[k]
            if j == 0:
                u = "c_{k-2}"
            elif j == 1:
                u = "c_{k-1}"
            else:
                u = str(j - 2)
            v = str(i)
            edges[u, v] = op_map[op]

    for i in range(steps):
        edges[str(i), "c_{k}"] = op_map['out']

    return ops, edges


def nb301_to_graph(n, net_key="net"):
    if not isinstance(n, str):
        n = n[net_key]

    genotype = convert_compact_to_genotype(eval(n))
    return darts_to_graph(genotype.normal), darts_to_graph(genotype.reduce)


def encode_to_onehot(net, benchmark):
    return onehot_conversions[benchmark](net)


def nb101_to_onehot(net):
    net = convert_tuple_to_spec(net)
    matrix_dim = len(net['matrix'])
    if matrix_dim < 7:
        padval = 7 - matrix_dim
        net['matrix'] = np.pad(net['matrix'], [(0, padval), (0, padval)])
        for _ in range(padval):
            net['ops'].insert(-1, 'maxpool3x3')

    enc = naslib.search_spaces.nasbench101.encodings.encode_adj(net)
    if matrix_dim < 7:
        for i in range(0, 7 - matrix_dim):
            for oid in range(3):
                idx = 3 * i + oid
                enc[-1 - idx] = 0
    return enc


bench_conversions = {
    'zc_nasbench101': nb101_to_graph,
    'zc_nasbench201': nb201_to_graph,
    'zc_nasbench301': nb301_to_graph,
    'zc_transbench101_micro': tnb101_to_graph
}


onehot_conversions = {
    'zc_nasbench101': nb101_to_onehot,
    'zc_nasbench201': encode_adjacency_one_hot_op_indices,
    'zc_nasbench301': naslib.search_spaces.nasbench301.encodings.encode_adj,
    'zc_transbench101_micro': encode_adjacency_one_hot_transbench_micro_op_indices
}
