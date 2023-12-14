import naslib
import networkx as nx
import numpy as np
import copy
from naslib.search_spaces.nasbench101.conversions import convert_tuple_to_spec

from naslib.search_spaces.nasbench201.encodings import encode_adjacency_one_hot_op_indices
from naslib.search_spaces.nasbench201.conversions import convert_str_to_op_indices, convert_op_indices_to_str, convert_op_indices_to_op_list

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
    # nodes_to_remove_list = []
    # remove_nodes_list = []
    edges_to_add_list = []
    for i, n in enumerate(node_labelling):
        G.nodes[i]['op_name'] = n
        # if n == 'none' or n == 'skip_connect':
        #     input_nodes = [edge[0] for edge in G.in_edges(i)]
        #     output_nodes = [edge[1] for edge in G.out_edges(i)]
        #     nodes_to_remove_info = {'id': i, 'input_nodes': input_nodes, 'output_nodes': output_nodes}
        #     nodes_to_remove_list.append(nodes_to_remove_info)
        #     remove_nodes_list.append(i)

            # if n == 'skip_connect':
            #     for n_i in input_nodes:
            #         edges_to_add = [(n_i, n_o) for n_o in output_nodes]
            #         edges_to_add_list += edges_to_add

    # # reconnect edges for removed nodes with 'skip_connect'
    # G.add_edges_from(edges_to_add_list)

    # # remove nodes with 'skip_connect' or 'none' label
    # G.remove_nodes_from(remove_nodes_list)

    # # after removal, some op nodes have no input nodes and some have no output nodes
    # # --> remove these redundant nodes
    # nodes_to_be_further_removed = []
    # for n_id in G.nodes():
    #     in_edges = G.in_edges(n_id)
    #     out_edges = G.out_edges(n_id)
    #     if n_id != 0 and len(in_edges) == 0:
    #         nodes_to_be_further_removed.append(n_id)
    #     elif n_id != 7 and len(out_edges) == 0:
    #         nodes_to_be_further_removed.append(n_id)

    # G.remove_nodes_from(nodes_to_be_further_removed)
    G.graph_type = 'node_attr'

    # create the arch string for querying nasbench dataset
    arch_query_string = f'|{op_node_labelling[0]}~0|+' \
                        f'|{op_node_labelling[1]}~0|{op_node_labelling[2]}~1|+' \
                        f'|{op_node_labelling[3]}~0|{op_node_labelling[4]}~1|{op_node_labelling[5]}~2|'

    G.name = arch_query_string
    return G


def prune(original_matrix, ops):
    """Prune the extraneous parts of the graph.

    General procedure:
      1) Remove parts of graph not connected to input.
      2) Remove parts of graph not connected to output.
      3) Reorder the vertices so that they are consecutive after steps 1 and 2.

    These 3 steps can be combined by deleting the rows and columns of the
    vertices that are not reachable from both the input and output (in reverse).
    """
    num_vertices = np.shape(original_matrix)[0]
    new_matrix = copy.deepcopy(original_matrix)
    new_ops = copy.deepcopy(ops)
    # DFS forward from input
    visited_from_input = {0}
    frontier = [0]
    while frontier:
        top = frontier.pop()
        for v in range(top + 1, num_vertices):
            if original_matrix[top, v] and v not in visited_from_input:
                visited_from_input.add(v)
                frontier.append(v)

    # DFS backward from output
    visited_from_output = {num_vertices - 1}
    frontier = [num_vertices - 1]
    while frontier:
        top = frontier.pop()
        for v in range(0, top):
            if original_matrix[v, top] and v not in visited_from_output:
                visited_from_output.add(v)
                frontier.append(v)

    # Any vertex that isn't connected to both input and output is extraneous to
    # the computation graph.
    extraneous = set(range(num_vertices)).difference(
        visited_from_input.intersection(visited_from_output))

    # If the non-extraneous graph is less than 2 vertices, the input is not
    # connected to the output and the spec is invalid.
    if len(extraneous) > num_vertices - 2:
        new_matrix = None
        new_ops = None
        valid_spec = False
        return

    new_matrix = np.delete(new_matrix, list(extraneous), axis=0)
    new_matrix = np.delete(new_matrix, list(extraneous), axis=1)
    for index in sorted(extraneous, reverse=True):
        del new_ops[index]

    return new_matrix, new_ops

def _preprocess(X, y=None):
    tmp = []
    valid_indices = []
    for idx, c in enumerate(X):
        node_labeling = list(nx.get_node_attributes(c, 'op_name').values())
        try:
            res = prune(nx.to_numpy_array(c), node_labeling)
            if res is None:
                continue
            c_new, label_new = res
            c_nx = nx.from_numpy_array(c_new, create_using=nx.DiGraph)
            for i, n in enumerate(label_new):
                c_nx.nodes[i]['op_name'] = n
        except KeyError:
            print('Pruning error!')
            c_nx = c
        tmp.append(c_nx)
        valid_indices.append(idx)
    if y is not None: y = y[valid_indices]
    if y is None:
        return tmp
    return tmp, y

def nb101_nx_graphs(net):
    net = convert_tuple_to_spec(net)
    nx_Graph = nx.from_numpy_array(net['matrix'], create_using=nx.DiGraph)
    nx_Graph.graph_type = 'node_attr'

    for i, n in enumerate(net['ops']):
        nx_Graph.nodes[i]['op_name'] = n


    # nx_Graph = _preprocess([nx_Graph])
    return nx_Graph

def nb201_nx_graphs(net):
    string = convert_op_indices_to_op_list(net)
    nx_Graph = create_nasbench201_graph(string)
    # nx_Graph = _preprocess([nx_Graph])
    return nx_Graph 

def nb301_nx_graphs(net):
    op_map = ['in', *get_ops_nb301(), 'out']
    op_map = {o: i for i, o in enumerate(op_map)}
    ops = {i: o for o, i in op_map.items()}
    dic = encode_gcn(net)
    matrix = dic['adjacency']
    ops_onehot = dic['operations'].nonzero()[1]
    # ops = [ops[i] for i in ops_onehot]

    nx_Graph = nx.from_numpy_array(matrix, create_using=nx.DiGraph)
    nx_Graph.graph_type = 'node_attr'

    for i, n in enumerate(ops_onehot):
        nx_Graph.nodes[i]['op_name'] = n

    # nx_Graph = _preprocess([nx_Graph])
    return nx_Graph

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