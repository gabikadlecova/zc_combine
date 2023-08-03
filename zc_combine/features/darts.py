import numpy as np

from zc_combine.features.base import count_ops, get_in_out_edges, max_num_on_path


# allowed ... tak jako sep zvlast, dil zvlast, kazdej sam, se skip connectionou


# TODO min path - nedava smysl, protoze jde z kazdyho nodu... takze spis max depth
#   - od inputu k jakýmukoli hlubšímu nodu, ne out

# TODO bridges? independent nodes? nejak ruzny topo sorty?


def _get_for_both_cells(cells, func):
    normal = func(cells[0])
    reduce = func(cells[1])

    if isinstance(normal, dict):
        return {**{f"normal_{k}": v for k, v in normal.items()}, **{f"reduce_{k}": v for k, v in reduce.items()}}

    return {'normal': normal, 'reduce': reduce}


def get_op_counts(cells, to_names=False):
    return _get_for_both_cells(cells, lambda c: count_ops(c, to_names=to_names))


def get_special_nodes():
    return ["c_{k-2}", "c_{k-1}", "out"]


def _get_cell_degrees(ops, edges, allowed):
    get_avg = lambda x: np.mean([len(v) for v in x.values()])

    input1, input2, output = get_special_nodes()

    in_edges, out_edges = get_in_out_edges(edges, allowed, start=0, end=len(ops))
    return {f'{input1}_degree': len(in_edges[input1]), f'{input2}_degree': len(in_edges[input2]),
            'out_degree': len(out_edges[output]), 'avg_in': get_avg(in_edges), 'avg_out': get_avg(out_edges)}


def get_node_degrees(cells, allowed):
    return _get_for_both_cells(cells, lambda c: _get_cell_degrees(*c, allowed))


def _get_both_paths(ops, edges, allowed):
    if 'out' not in allowed:
        if isinstance(allowed, set):
            allowed.add('out')
        else:
            allowed = [*allowed, 'out']

    input1, input2, output = get_special_nodes()

    res1 = max_num_on_path((ops, edges), allowed, start=input1, end=output)
    res2 = max_num_on_path((ops, edges), allowed, start=input2, end=output)
    return {f"{input1}": res1, f"{input2}": res2}


def get_max_path(cells, allowed):
    return _get_for_both_cells(cells, lambda c: _get_both_paths(*c, allowed))


def min_node_path(cells, banned):
    """Compared to min path len in nb201, every node is connected to the output and so we look at intermediate nodes
       to assess depth."""
    pass


feature_func_dict = {
    'op_count': count_ops,
    'node_degree': get_node_degrees,
    'max_op_on_path': get_max_path
}
