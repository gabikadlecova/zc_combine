import json
import os.path

import numpy as np
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


def keep_unique_nets(data, tnb=False, filter_nets=None):
    _, edge_map = get_ops_edges_tnb101() if tnb else get_ops_edges_nb201()
    nb201_zero_out_unreachable(data, edge_map, zero_op=0 if tnb else 1)

    if filter_nets is not None:
        data = keep_only_isomorpic_nb201(data, filter_nets, zero_is_1=not tnb, net_key='new_net')
    return data[data['net'] == data['new_net']]


def load_bench_data(searchspace_path, benchmark, dataset, filter_nets=None, zero_op_filtering=False):
    benchmark = get_bench_key(benchmark)
    search_space = load_search_space(searchspace_path, benchmark)
    dfs = parse_scores(search_space)
    data = dfs[dataset]

    tnb = 'trans' in benchmark
    if tnb or '201' in benchmark and zero_op_filtering:
        data = keep_unique_nets(data, tnb=tnb, filter_nets=filter_nets)

    return data


def get_net_data(data, benchmark, net_str='net'):
    benchmark = get_bench_key(benchmark)
    convert_func = bench_conversions[benchmark]

    return {i: convert_func(data.loc[i], net_key=net_str) for i in data.index}


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

    res_data.columns = [c.replace('[', '(').replace(']', ')') for c in res_data.columns]
    return res_data, y


def get_data_splits(data, y, **kwargs):
    tr_X, te_X, tr_y, te_y = train_test_split(data, y, **kwargs)
    return {"train_X": tr_X, "test_X": te_X, "train_y": tr_y, "test_y": te_y}


def _dict_to_lists(mdict, res_dict, prefix='', mean_m=False):
    for k, m in mdict.items():
        mlist = res_dict.setdefault(f"{k}{prefix}", [])

        if mean_m:
            if not isinstance(m, float):
                m = m[0] if len(m) == 1 else np.mean(m)
        mlist.append(m)


def eval_model(model_cls, data, n_times=1, random_state=43, subsample=True, sample_size=200, sample_times=10):
    res = {'seed': []}
    models = []

    for i in range(n_times):
        res['seed'].append(random_state + i)
        model = model_cls(random_state + i)
        model.fit(data['train_X'], data['train_y'])

        def save_preds(sample=False):
            res_metrics = {}
            repeats = sample_times if sample else 1

            # sample n times
            for j in range(repeats):
                metrics = predict_on_test(model, data['test_X'], data['test_y'], sample=sample_size if sample else None,
                                          seed=random_state + j)
                _dict_to_lists(metrics, res_metrics)

            # save mean of sampled
            _dict_to_lists(res_metrics, res, prefix='_sample' if sample else '', mean_m=True)

        save_preds()
        if subsample:
            save_preds(True)
        models.append(model)

    return models, res


def predict_on_test(model, test_X, test_y, sample=None, seed=None):
    res = {}

    if sample is not None:
        state = np.random.RandomState(seed) if seed is not None else np.random
        idxs = state.randint(0, len(test_y), sample)
        test_X, test_y = test_X.iloc[idxs], test_y.iloc[idxs]

    preds = model.predict(test_X)
    true = test_y

    res['r2'] = r2_score(true, preds)
    res['mse'] = mean_squared_error(true, preds)
    res['tau'] = kendalltau(preds, true)[0]
    res['corr'] = spearmanr(preds, true)[0]
    return res


def parse_columns_filename(path):
    path, _ = os.path.splitext(os.path.basename(path))

    args = path.split('-')
    mode_idx = 'idx' in path
    has_prefix = 5 if mode_idx else 4
    has_prefix = len(args) == has_prefix

    names = ['prefix'] if has_prefix else []
    names = ['name', *names, 'n_features', 'mode']
    names = [*names, 'idx'] if mode_idx else names
    return {f"cols_{n}": v for n, v in zip(names, args)}
