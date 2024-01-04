import naslib
import networkx as nx
import numpy as np
import copy
from naslib.search_spaces.nasbench101.conversions import convert_tuple_to_spec

from naslib.search_spaces.nasbench201.conversions import convert_str_to_op_indices, convert_op_indices_to_str, \
    OP_NAMES_NB201, EDGE_LIST
from naslib.search_spaces.nasbench201.encodings import encode_adjacency_one_hot_op_indices

from naslib.search_spaces.nasbench301.conversions import convert_compact_to_genotype
from naslib.search_spaces.nasbench301.encodings import encode_adj, encode_gcn

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

def nb201_arch2vec_embedding(net, embedding_data):
    string = convert_op_indices_to_str(net)
    embedding_data_ops = [v['ops'] for v in embedding_data.values()]
    keys = embedding_data[embedding_data_ops.index(string)]
    embedding = keys['feature'].detach().tolist()
    return embedding


def nb101_arch2vec_embedding(net, embedding_data):
    net = convert_tuple_to_spec(net)
    keys = [k for k,v in embedding_data.items() if v['adj'][:len(v['ops']), :len(v['ops'])].flatten().tolist()==net['matrix'].flatten().tolist() and v['ops']==net['ops']]
    embedding = embedding_data[keys[0]]['feature'].detach().tolist()
    return embedding


# === For NASBench-201 ====
def create_nasbench201_graph(op_node_labelling, edge_attr=False):
    assert len(op_node_labelling) == 6
    # the graph has 8 nodes (6 operation nodes + input + output)
    G = nx.DiGraph()
    edge_list = [(0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 6), (3, 6), (4, 7), (5, 7), (6, 7)]
    G.add_edges_from(edge_list)

    # assign node attributes and collate the information for nodes to be removed
    # (i.e. nodes with 'skip_connect' or 'none' label)
    node_labelling = ['input'] + op_node_labelling + ['output']
    edges_to_add_list = []
    for i, n in enumerate(node_labelling):
        G.nodes[i]['op_name'] = n

    # G.remove_nodes_from(nodes_to_be_further_removed)
    G.graph_type = 'node_attr'

    # create the arch string for querying nasbench dataset
    arch_query_string = f'|{op_node_labelling[0]}~0|+' \
                        f'|{op_node_labelling[1]}~0|{op_node_labelling[2]}~1|+' \
                        f'|{op_node_labelling[3]}~0|{op_node_labelling[4]}~1|{op_node_labelling[5]}~2|'

    G.name = arch_query_string
    return G

def nb101_nx_graphs(net):
    net = convert_tuple_to_spec(net)
    nx_Graph = nx.from_numpy_array(net['matrix'], create_using=nx.DiGraph)
    nx_Graph.graph_type = 'node_attr'

    for i, n in enumerate(net['ops']):
        nx_Graph.nodes[i]['op_name'] = n

    return nx_Graph


def _convert_op_indices_to_op_list(op_indices):
    edge_op_dict = {
        edge: OP_NAMES_NB201[op] for edge, op in zip(EDGE_LIST, op_indices)
    }

    op_edge_list = [
        "{}".format(edge_op_dict[(i, j)])
        for i, j in sorted(edge_op_dict, key=lambda x: x[1])
    ]
    return op_edge_list


def nb201_nx_graphs(net):
    string = _convert_op_indices_to_op_list(net)
    nx_Graph = create_nasbench201_graph(string)
    return nx_Graph 


def nb301_nx_graphs(net):
    op_map = ['in', *get_ops_nb301(), 'out']
    op_map = {o: i for i, o in enumerate(op_map)}
    ops = {i: o for o, i in op_map.items()}
    dic = encode_gcn(net)
    matrix = dic['adjacency']
    ops_onehot = dic['operations'].nonzero()[1]

    nx_Graph = nx.from_numpy_array(matrix, create_using=nx.DiGraph)
    nx_Graph.graph_type = 'node_attr'

    for i, n in enumerate(ops_onehot):
        nx_Graph.nodes[i]['op_name'] = n

    return nx_Graph

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
    'zc_nasbench301': encode_adj,
    'zc_transbench101_micro': encode_adjacency_one_hot_transbench_micro_op_indices,
    'zc_transbench101_macro': encode_adjacency_one_hot_transbench_macro_op_indices
}


embedding_conversions = {
    'zc_nasbench101': nb101_arch2vec_embedding, 
    'zc_nasbench201': nb201_arch2vec_embedding
}


wl_feature_conversions = {
    'zc_nasbench101': nb101_nx_graphs, 
    'zc_nasbench201': nb201_nx_graphs, 
    'zc_nasbench301': nb301_nx_graphs
}


path_conversions = {
    'zc_nasbench101': nb101_to_paths,
    'zc_nasbench201': nb201_to_paths,
    'zc_nasbench301': naslib.search_spaces.nasbench301.encodings.encode_paths
}

