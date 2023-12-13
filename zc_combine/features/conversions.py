import naslib
import networkx as nx
import numpy as np
from naslib.search_spaces.nasbench101.conversions import convert_tuple_to_spec
from naslib.search_spaces.nasbench201.encodings import encode_adjacency_one_hot_op_indices, encode_paths

from naslib.search_spaces.nasbench301.conversions import convert_compact_to_genotype
from naslib.search_spaces.nasbench201.conversions import convert_str_to_op_indices
from naslib.search_spaces.transbench101.encodings import encode_adjacency_one_hot_transbench_micro_op_indices
from naslib.search_spaces.transbench101.encodings import encode_adjacency_one_hot_transbench_macro_op_indices

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


def tnb101_macro_encode(net, net_key='net'):
    if not isinstance(net, str):
        net = net[net_key]

    res = []
    net = net.strip('()').split(',')

    for idx in net:
        encoding = {}
        idx = int(idx)
        encoding['channel'] = idx % 2 == 0
        encoding['stride'] = idx > 2
        res.append(encoding)

    return res


def encode_to_onehot(net, benchmark):
    return onehot_conversions[benchmark](net)


def pad_nb101_net(net):
    matrix_dim = len(net['matrix'])
    if matrix_dim < 7:
        padval = 7 - matrix_dim
        net['matrix'] = np.pad(net['matrix'], [(0, padval), (0, padval)])
        for _ in range(padval):
            net['ops'].insert(-1, 'maxpool3x3')

    return net


def nb101_to_onehot(net):
    net = convert_tuple_to_spec(net)
    matrix_dim = len(net['matrix'])
    net = pad_nb101_net(net)

    enc = naslib.search_spaces.nasbench101.encodings.encode_adj(net)
    if matrix_dim < 7:
        for i in range(0, 7 - matrix_dim):
            for oid in range(3):
                idx = 3 * i + oid
                enc[-1 - idx] = 0
    return enc


def nb101_to_paths(net):
    net = convert_tuple_to_spec(net)
    net = pad_nb101_net(net)
    return naslib.search_spaces.nasbench101.encodings.encode_paths(net)


def get_paths(arch):
    """
    return all paths from input to output
    """
    path_blueprints = [[3], [0, 4], [1, 5], [0, 2, 5]]
    paths = []
    for blueprint in path_blueprints:
        paths.append([arch[node] for node in blueprint])
    return paths


def get_path_indices(arch, num_ops=5):
    """
    compute the index of each path
    """
    paths = get_paths(arch)
    path_indices = []

    for i, path in enumerate(paths):
        if i == 0:
            index = 0
        elif i in [1, 2]:
            index = num_ops
        else:
            index = num_ops + num_ops ** 2
        for j, op in enumerate(path):
            index += op * num_ops ** j
        path_indices.append(index)

    return tuple(path_indices)


def nb201_to_paths(net, num_ops=5, longest_path_length=3):
    # FROM NASLIB #
    num_paths = sum([num_ops ** i for i in range(1, longest_path_length + 1)])

    path_indices = get_path_indices(net, num_ops=num_ops)

    encoding = np.zeros(num_paths)
    for index in path_indices:
        encoding[index] = 1
    return encoding


bench_conversions = {
    'zc_nasbench101': nb101_to_graph,
    'zc_nasbench201': nb201_to_graph,
    'zc_nasbench301': nb301_to_graph,
    'zc_transbench101_micro': tnb101_to_graph,
    'zc_transbench101_macro': tnb101_macro_encode
}


onehot_conversions = {
    'zc_nasbench101': nb101_to_onehot,
    'zc_nasbench201': encode_adjacency_one_hot_op_indices,
    'zc_nasbench301': naslib.search_spaces.nasbench301.encodings.encode_adj,
    'zc_transbench101_micro': encode_adjacency_one_hot_transbench_micro_op_indices,
    'zc_transbench101_macro': encode_adjacency_one_hot_transbench_macro_op_indices
}


path_conversions = {
    'zc_nasbench101': nb101_to_paths,
    'zc_nasbench201': nb201_to_paths,
    'zc_nasbench301': naslib.search_spaces.nasbench301.encodings.encode_paths
}
