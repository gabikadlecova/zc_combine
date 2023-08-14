import numpy as np

from zc_combine.features.base import count_ops_opnodes, min_path_len_opnodes, max_num_on_path_opnodes, get_start_end, \
    get_in_out_edges_opnodes


def node_degree(net, allowed, start=0, end=1):
    _, ops, graph = net
    in_edges, out_edges = get_in_out_edges_opnodes(net, allowed)

    start, end = get_start_end(ops, start, end)
    get_avg = lambda x: np.mean([len(v) for v in x.values()])

    return {'in_degree': len(in_edges[start]), 'out_degree': len(out_edges[end]), 'avg_in': get_avg(in_edges),
            'avg_out': get_avg(out_edges)}


feature_func_dict = {
    'op_count': count_ops_opnodes,
    'min_path_len': min_path_len_opnodes,
    'max_op_on_path': max_num_on_path_opnodes,
    'node_degree': node_degree
}
