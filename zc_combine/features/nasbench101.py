import networkx as nx
import numpy as np

from zc_combine.features.base import count_ops_opnodes, min_path_len_opnodes, max_num_on_path_opnodes, get_start_end, \
    get_in_out_edges_opnodes


def node_degree(net, allowed, start=0, end=1):
    _, ops, graph = net
    in_edges, out_edges = get_in_out_edges_opnodes(net, allowed)

    start, end = get_start_end(ops, start, end)
    get_func = lambda x, y: y([len(v) for v in x.values()])

    return {'in_degree': len(in_edges[end]), 'out_degree': len(out_edges[start]), 'avg_in': get_func(in_edges, np.mean),
            'avg_out': get_func(out_edges, np.mean), 'max_out': get_func(out_edges, np.max),
            'max_in': get_func(in_edges, np.max)}


def num_of_paths(net, start=0, end=1):
    _, ops, graph = net

    start, end = get_start_end(ops, start, end)
    path = nx.all_simple_paths(graph, start, end)
    return len([p for p in path])


feature_func_dict = {
    'op_count': count_ops_opnodes,
    'min_path_len': min_path_len_opnodes,
    'max_op_on_path': max_num_on_path_opnodes,
    'node_degree': node_degree,
    'num_paths': num_of_paths
}
