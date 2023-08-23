import json

import pandas as pd
from scipy.stats import kendalltau, spearmanr
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

from zc_combine.features import feature_dicts
from zc_combine.features.conversions import keep_only_isomorpic_nb201, bench_conversions
from zc_combine.features.dataset import get_feature_dataset
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


def get_dataset(data, nets, benchmark, cfg=None, features=None, proxy_cols=None, use_features=True,
                use_all_proxies=False, use_flops_params=True):
    # compute network features
    feature_dataset = []
    if use_features:
        assert cfg is not None, "Must provide config when using network features."
        feature_dataset = get_feature_dataset(nets, cfg, feature_dicts[get_bench_key(benchmark)], subset=features)
        feature_dataset = [f for f in feature_dataset.values()]

    if use_all_proxies:
        proxy_cols = set(c for c in data.columns if c not in ['random', 'rank', 'new_net', 'net'])
    else:
        proxy_cols = proxy_cols if proxy_cols is not None else []
        proxy_cols = {'flops', 'params', *proxy_cols} if use_flops_params else set(proxy_cols)
    proxy_df = data[[c for c in data.columns if c in proxy_cols or c == 'val_accs']]

    # get data and y
    res_data = pd.concat([*feature_dataset, proxy_df], axis=1)
    y = res_data['val_accs']
    res_data.drop(columns=['val_accs'], inplace=True)

    return res_data, y


def get_data_splits(data, y, **kwargs):
    tr_X, te_X, tr_y, te_y = train_test_split(data, y, **kwargs)
    return {"train_X": tr_X, "test_X": te_X, "train_y": tr_y, "test_y": te_y}


def eval_model(model_cls, data, n_times=1, random_state=43):
    res = {'r2': [], 'mse': [], 'tau': [], 'corr': []}
    models = []

    for i in range(n_times):
        model = model_cls(random_state + i)
        model.fit(data['train_X'], data['train_y'])

        preds = model.predict(data['test_X'])
        true = data['test_y']
        res['r2'].append(r2_score(true, preds))
        res['mse'].append(mean_squared_error(true, preds))
        res['tau'].append(kendalltau(preds, true)[0])
        res['corr'].append(spearmanr(preds, true)[0])
        models.append(model)

    return models, res
