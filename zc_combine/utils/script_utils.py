import json
import os.path
import pickle
import pdb

import numpy as np
import pandas as pd
import time
from scipy.stats import kendalltau, spearmanr
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import torch
from sklearn.preprocessing import StandardScaler

from zc_combine.features import feature_dicts
from zc_combine.features.conversions import keep_only_isomorpic_nb201, bench_conversions, onehot_conversions, \
    embedding_conversions, wl_feature_conversions, path_conversions
from zc_combine.features.dataset import get_feature_dataset
from zc_combine.fixes.operations import get_ops_edges_nb201, get_ops_edges_tnb101
from zc_combine.fixes.utils import nb201_zero_out_unreachable
from zc_combine.utils.naslib_utils import load_search_space, parse_scores

from zc_combine.features.wl_kernel.kernels.weisfeilerlehman import WeisfeilerLehman


def load_feature_proxy_dataset(searchspace_path, benchmark, dataset, cfg=None, features=None, proxy=None, meta=None,
                               use_features=True, use_all_proxies=False, use_flops_params=True, use_onehot=False,
                               use_embedding=False, use_path_encoding=False, zero_unreachable=True, keep_uniques=True,
                               target_csv=None, target_key='val_accs', cache_path=None, version_key=None,
                               compute_all=False, embedding_path='../data/arch2vec/', multi_objective=False):
    """
        Load feature and proxy datasets, feature dataset can be precomputed or will be loaded from the config.
        Validation accuracy will be returned as the target.
    Args:
        searchspace_path: Path where zero-cost proxy json files from NASLib are saved.
        benchmark: One of NAS benchmarks (look at `bench_names` in this file for list of supported benches).
        dataset: Datasets precomputed on benchmarks
        cfg: Config for computing simple features.
        features: Subset of feature types to use.
        proxy: Subset of proxies to use.
        meta: Json with unique nb201 networks for filtering (can be used for tnb101 too).
        use_features: If True, features will be included.
        use_all_proxies: If True, include all proxies regardless of other settings.
        use_flops_params: If True, flops and params will always be included.
        use_onehot: If True, add onehot encoding of the architecture.
        use_embedding: If True, add the precomputed arch2vec embedding of the architecture.
        use_path_encoding: If True, add path encoding of the architecture (all possible input-output paths).
        zero_unreachable: If True, keep only networks with no unreachable operations. Does not lead to all uniques,
            as there are also isomorphisms due to skip connections.

        keep_uniques: If False, keep all networks.
        target_csv: If None, val_accs from the proxy json is used. Otherwise, `target_key` column from the loaded csv.
        target_key: Target key to predict (and extract from the `target_csv`).
        cache_path: Path to either save the feature dataset, or to load from.
        version_key: Version key to check with loaded dataset, or for saving it.
        compute_all: If True, compute all features even if `features` is None. Good for caching.
        embedding_path: Path to saved arch2vec embeddings.
        multi_objective: If only target_key or several targets from csv file

    Returns:
        dataset, y - feature and/or proxy dataset, validation accuracy
    """
    # load networks, for nb201 and tnb101, drop duplicates if `keep_uniques`
    meta = meta if keep_uniques else None
    if meta is not None:
        with open(meta, 'r') as f:
            meta = json.load(f)

    zero_unreachable = zero_unreachable if keep_uniques else False
    data = load_bench_data(searchspace_path, benchmark, dataset, filter_nets=meta, zero_unreachable=zero_unreachable)

    if cfg is not None:
        with open(cfg, 'r') as f:
            cfg = json.load(f)

    features = features if features is None else features.split(',')
    proxy = proxy.split(',') if proxy is not None else []

    y = get_accs_or_target(data, target_csv=target_csv, target_key=target_key, multi_objective=multi_objective)

    nets = data['net']
    proxy_cols = _get_proxy_columns(data, use_all_proxies, use_flops_params, proxy)

    data = get_dataset(data, benchmark, cfg=cfg, features=features, proxy_cols=proxy,
                       use_features=use_features, use_all_proxies=use_all_proxies, use_flops_params=use_flops_params,
                       use_onehot=use_onehot, use_embedding=use_embedding, use_path_encoding=use_path_encoding,
                       cache_path=cache_path, version_key=version_key, compute_all=compute_all,
                       embedding_path=embedding_path, multi_objective=multi_objective)
    return {'nets': nets, 'proxy_columns': proxy_cols}, data, y


bench_names = {
    'nb101': 'zc_nasbench101',
    'nb201': 'zc_nasbench201',
    'nb301': 'zc_nasbench301',
    'tnb101': 'zc_transbench101_micro',
    'tnb101_macro': 'zc_transbench101_macro'
}


def get_accs_or_target(data, target_csv=None, target_key=None, multi_objective=None):
    # either use validation accuracy as the target, or a user-defined metric in a separate csv file
    if target_csv is None:
        y = data[target_key]
    else:
        target_df = pd.read_csv(target_csv)
        y = get_target(target_df, data['net'], target_key=target_key, multi_objective=multi_objective)
    return y


def create_cache_filename(cache_dir, cfg_path, features, version_key, compute_all):
    assert os.path.isdir(cache_dir)

    cfg_name = os.path.splitext(os.path.basename(cfg_path))[0]
    # for compute all, all features are always computed and cached
    if not compute_all:
        features = '' if features is None else '_'.join(sorted(features.split(',')))
        features = f'-{features}' if len(features) else features
    else:
        features = ''

    version_key = f'-{version_key}' if version_key is not None and len(version_key) else version_key
    return os.path.join(cache_dir, f'{cfg_name}{features}{version_key}.pickle')


def get_bench_key(benchmark):
    return bench_names[benchmark] if benchmark in bench_names else benchmark


def keep_unique_nets(data, tnb=False, filter_nets=None, zero_unreachable=True):
    _, edge_map = get_ops_edges_tnb101() if tnb else get_ops_edges_nb201()

    if zero_unreachable:
        # create column 'new_net' - isomorphic to 'net', but unreachable ops (due to zero ops) are zeroed out
        nb201_zero_out_unreachable(data, edge_map, zero_op=0 if tnb else 1)

    if filter_nets is not None:
        # filter out rows corresponding to unique nets
        unique_data = keep_only_isomorpic_nb201(data, filter_nets, zero_is_1=not tnb, net_key='net', copy=False)

        # return rows where 'net' is same as uniques' 'new_net' (unique 'net' may include nets without unreachables zeroed out)
        unique_new_nets = unique_data['new_net' if zero_unreachable else 'net']
        return data[data['net'].isin(unique_new_nets)]

    return data[data['net'] == data['new_net']] if zero_unreachable else data


def load_bench_data(searchspace_path, benchmark, dataset, filter_nets=None, zero_unreachable=True):
    benchmark = get_bench_key(benchmark)
    search_space = load_search_space(searchspace_path, benchmark)
    dfs = parse_scores(search_space)
    data = dfs[dataset]

    tnb = 'trans' in benchmark and 'macro' not in benchmark
    if tnb or '201' in benchmark:
        data = keep_unique_nets(data, tnb=tnb, filter_nets=filter_nets, zero_unreachable=zero_unreachable)

    return data


def get_net_data(data, benchmark, net_str='net'):
    benchmark = get_bench_key(benchmark)
    convert_func = bench_conversions[benchmark]

    return {i: convert_func(data.loc[i], net_key=net_str) for i in data.index}


def load_or_create_features(nets, cfg, benchmark, features=None, cache_path=None, version_key=None, compute_all=False):
    if cache_path is not None and os.path.exists(cache_path):
        # load from cache path, check if version keys are the same
        with open(cache_path, 'rb') as f:
            cached_data = pickle.load(f)
        assert version_key is None or version_key == cached_data['version_key']

        feature_dataset = cached_data['dataset']
    else:
        # create features when no cache path
        assert cfg is not None, "Must provide config when using network features."
        subset = None if compute_all else features
        feature_dataset = get_feature_dataset(nets, cfg, feature_dicts[get_bench_key(benchmark)], subset=subset)

    # save to cache path if features were just created
    if cache_path is not None and not os.path.exists(cache_path):
        save_data = {'dataset': feature_dataset, 'version_key': version_key}
        with open(cache_path, 'wb') as f:
            pickle.dump(save_data, f)

    if features is not None and compute_all:
        feature_dataset = [f for k, f in feature_dataset.items() if k in features]
    else:
        feature_dataset = [f for f in feature_dataset.values()]

    return feature_dataset


def _get_proxy_columns(data, use_all_proxies, use_flops_params, proxy_cols):
    if use_all_proxies:
        proxy_cols = set(c for c in data.columns if c not in ['random', 'rank', 'new_net', 'net', 'val_accs'])
    else:
        proxy_cols = proxy_cols if proxy_cols is not None else []
        proxy_cols = {'flops', 'params', *proxy_cols} if use_flops_params else set(proxy_cols)
    return proxy_cols


def get_dataset(data, benchmark, cfg=None, features=None, proxy_cols=None, use_features=True, use_all_proxies=False,
                use_onehot=False, use_embedding=False, use_flops_params=True, use_path_encoding=False, cache_path=None, version_key=None,
                compute_all=False, embedding_path='../data/arch2vec/', multi_objective=False):
    feature_dataset = []
    # compute or load network features
    if use_features:
        nets = get_net_data(data, benchmark)
        feature_dataset = load_or_create_features(nets, cfg, benchmark, features=features, cache_path=cache_path,
                                                  version_key=version_key, compute_all=compute_all)

    proxy_cols = _get_proxy_columns(data, use_all_proxies, use_flops_params, proxy_cols)
    proxy_df = data[[c for c in data.columns if c in proxy_cols]]

    onehot = []
    if use_onehot:
        onehot.append(get_onehot_encoding(data, benchmark))

    embedding_features = []
    if use_embedding:
        embedding_features.append(get_embedding(data, benchmark, embedding_path=embedding_path))
        
    path_enc = []
    if use_path_encoding:
        path_enc.append(get_path_encoding(data, benchmark))

    # Add net string back here for WL kernel calculation
    # get data and y
    res_data = pd.concat([*feature_dataset, proxy_df, *onehot, *path_enc, *embedding_features, data['net']],
                         axis=1)
    res_data.columns = [c.replace('[', '(').replace(']', ')') for c in res_data.columns]
    if 'val_accs' in res_data.columns:
        res_data.drop(columns='val_accs', inplace=True)
    return res_data


def get_target(target_data, net_tuples, target_key='val_accs', net_key='net', multi_objective=None):
    # select nets based on net_tuples
    target_data = target_data.reset_index(drop=True).set_index(net_key)
    target_data = target_data.loc[net_tuples].reset_index().set_index(net_tuples.index).rename_axis(None)
    
    if multi_objective:
        # forcely include also val_accs as target_key
        return target_data[['val_accs', target_key]]
    else:
        return target_data[target_key]


def get_embedding(data, benchmark, embedding_path='../data/arch2vec/'):
    cache_embedding_path = os.path.join(str(embedding_path), str(benchmark)+"_embeddings.csv")
    if not os.path.exists(cache_embedding_path):
        print('create embedding data')
        embedding_data = torch.load(embedding_path + str(benchmark) + '_embeddings.pt')
        embedding_convert = embedding_conversions[get_bench_key(benchmark)]
        embeddings = {i: embedding_convert(eval(data.loc[i]['net']), embedding_data) for i in data.index}
        embeddings = pd.DataFrame(embeddings.values(), index=embeddings.keys())
        embeddings.columns = [f"embeddings_{c}" for c in embeddings.columns]
        embeddings = pd.concat([data['net'], embeddings], axis=1)
        embeddings.to_csv(cache_embedding_path)
    else: 
        print('load embedding data')
        embeddings = pd.read_csv(cache_embedding_path, index_col=0)
    return embeddings


def get_wl_embedding(data, benchmark, key='net'):
    train_data = data['train_X']
    test_data = data['test_X']

    wl_feature_convert = wl_feature_conversions[get_bench_key(benchmark)]

    list_nx_graphs = [wl_feature_convert(eval(train_data.iloc[i][key])) for i in range(len(train_data))]
    kernel = WeisfeilerLehman(oa=False, h=1, requires_grad=True)
    kernel.fit_transform(list_nx_graphs)
    
    feat_list = kernel.feature_value(list_nx_graphs)[0].tolist()
    wl_features = {i: feat_list[i] for i in range(len(train_data))}
    wl_features = pd.DataFrame(wl_features.values(), index=wl_features.keys())
    wl_features.columns = [f"wl_feat_{c}" for c in wl_features.columns]
    data['train_X'] = pd.concat([data['train_X'], pd.DataFrame(columns = wl_features.columns)], axis=1)
    data['train_X'][wl_features.columns] = wl_features.values

    list_nx_graphs_test = [wl_feature_convert(eval(test_data.iloc[i][key])) for i in range(len(test_data))]

    feat_list_test = kernel.feature_value(list_nx_graphs_test)[0].tolist()

    wl_features_test = {i: feat_list_test[i] for i in  range(len(test_data))}
    wl_features_test = pd.DataFrame(wl_features_test.values(), index=wl_features_test.keys())
    wl_features_test.columns = [f"wl_feat_{c}" for c in wl_features_test.columns]

    data['test_X'] = pd.concat([data['test_X'], pd.DataFrame(columns = wl_features_test.columns)], axis=1)
    data['test_X'][wl_features_test.columns] = wl_features_test.values

    return data


def get_other_encoding(data, encode_func, prefix, net_key='net'):
    enc = {i: encode_func(eval(data.loc[i][net_key])) for i in data.index}
    enc = pd.DataFrame(enc.values(), index=enc.keys())
    enc.columns = [f"{prefix}_{c}" for c in enc.columns]
    return enc


def get_onehot_encoding(data, benchmark, net_key='net'):
    onehot_convert = onehot_conversions[get_bench_key(benchmark)]
    return get_other_encoding(data, onehot_convert, 'onehot', net_key=net_key)


def get_path_encoding(data, benchmark, net_key='net'):
    path_convert = path_conversions[get_bench_key(benchmark)]
    return get_other_encoding(data, path_convert, 'path', net_key=net_key)


def get_data_splits(data, y, **kwargs):
    tr_X, te_X, tr_y, te_y = train_test_split(data, y, **kwargs)
    return {"train_X": tr_X, "test_X": te_X, "train_y": tr_y, "test_y": te_y}


def normalize_columns(data_splits, proxy_cols):
    normalizer = StandardScaler()
    normalizer.fit(data_splits['train_X'][proxy_cols])
    train_proxies = normalizer.transform(data_splits['train_X'][proxy_cols])
    test_proxies = normalizer.transform(data_splits['test_X'][proxy_cols])

    for i, c in enumerate(data_splits['train_X'].columns):
        assert data_splits['test_X'].columns[i] == c

    for i, p in zip(range(train_proxies.shape[1]), proxy_cols):
        data_splits['train_X'][p] = train_proxies[:, i]
        data_splits['test_X'][p] = test_proxies[:, i]

    return data_splits


def _dict_to_lists(mdict, res_dict, prefix='', mean_m=False):
    for k, m in mdict.items():
        mlist = res_dict.setdefault(f"{k}{prefix}", [])

        if mean_m:
            if not isinstance(m, float):
                m = m[0] if len(m) == 1 else np.mean(m)
        mlist.append(m)


def eval_model(model_cls, data, n_times=1, random_state=43, subsample=True, sample_size=200, sample_times=10):
    res = {'seed': [], 'fit_time': []}
    models = []

    for i in range(n_times):
        res['seed'].append(random_state + i)
        model = model_cls(random_state + i)

        start = time.time()
        model.fit(data['train_X'], data['train_y'])
        res['fit_time'].append(time.time() - start)

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

    start = time.time()
    preds = model.predict(test_X)
    res['test_time'] = time.time() - start

    true = test_y

    res['r2'] = r2_score(true, preds)
    res['mse'] = mean_squared_error(true, preds)
    if len(preds.shape) == 2:
        res['corr'] = np.mean([spearmanr(preds[:,0], true.iloc[:,0])[0], spearmanr(preds[:,1], true.iloc[:,1])[0]])
        res['tau'] = np.mean([kendalltau(preds[:,0], true.iloc[:,0])[0], kendalltau(preds[:,1], true.iloc[:,1])[0]])
    else:
        res['corr'] = spearmanr(preds, true)[0]
        res['tau'] = kendalltau(preds, true)[0]
    return res
