import json

from zc_combine.features.conversions import keep_only_isomorpic_nb201, bench_conversions
from zc_combine.fixes.operations import get_ops_edges_nb201, get_ops_edges_tnb101
from zc_combine.fixes.utils import nb201_zero_out_unreachable
from zc_combine.utils.naslib_utils import load_search_space, parse_scores


bench_names = {
    'nb101': 'zc_nasbench101',
    'nb201': 'zc_nasbench201',
    'nb301': 'zc_nasbench301',
    'tnb101': 'zc_transbench101_micro'
}


def get_bench_key(benchmark):
    return bench_names[benchmark] if benchmark in bench_names else benchmark


def load_uniques(meta_path):
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    return [v['nb201-string'] for k, v in meta['ids'].items() if k == v['isomorph']]


def keep_unique_nets(data, tnb=False, filter_nets=None):
    _, edge_map = get_ops_edges_tnb101() if tnb else get_ops_edges_nb201()
    nb201_zero_out_unreachable(data, edge_map, zero_op=0 if tnb else 1)

    if filter_nets is not None:
        data = keep_only_isomorpic_nb201(data, filter_nets, zero_is_1=not tnb, net_key='new_net')
    return data[data['net'] == data['new_net']]


def load_bench_data(searchspace_path, benchmark, dataset, filter_nets=None):
    benchmark = get_bench_key(benchmark)
    search_space = load_search_space(searchspace_path, benchmark)
    dfs = parse_scores(search_space)
    data = dfs[dataset]

    tnb = 'trans' in benchmark
    if tnb or '201' in benchmark:
        data = keep_unique_nets(data, tnb=tnb, filter_nets=filter_nets)

    return data


def get_net_data(data, benchmark, net_str='net'):
    benchmark = get_bench_key(benchmark)
    convert_func = bench_conversions[benchmark]

    return {i: convert_func(data.loc[i], net_str) for i in data.index}
