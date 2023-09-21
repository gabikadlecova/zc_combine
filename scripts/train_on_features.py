import os.path
import click
import json
import numpy as np
import pandas as pd
import time

import scipy.stats
import seaborn as sns
import matplotlib.pyplot as plt

from datetime import datetime

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from utils import load_bench_data, get_net_data, get_dataset, get_data_splits, eval_model, parse_columns_filename
from zc_combine.predictors import predictor_cls


def do_pca(fit_data, transform_data, transform_y, n_components, compute_loadings=True, standardize=True):
    pca = PCA(n_components=n_components)

    if standardize:
        scaler = StandardScaler()
        pca.fit(scaler.fit_transform(fit_data))
        pca_data = pca.transform(scaler.transform(transform_data))
    else:
        pca.fit(fit_data)
        pca_data = pca.transform(transform_data)

    plot_dd = pd.DataFrame(pca_data)
    plot_dd['val_accs'] = pd.qcut(transform_y, 4)
    indices = {v: i for i, v in enumerate(sorted(plot_dd['val_accs'].unique()))}
    plot_dd['val_accs_idx'] = plot_dd['val_accs'].map(indices).astype(int)

    importances = (pca.components_.T * np.sqrt(pca.explained_variance_)).T if compute_loadings else pca.components_
    pca_importances = pd.DataFrame(data=importances, columns=fit_data.columns)
    return plot_dd, pca_importances


def _plot_pca(pca_data, title, save):
    pca_features, pca_loadings = pca_data

    sns.set()

    plt.figure(figsize=(10, 8))
    ax = sns.scatterplot(data=pca_features, x=0, y=1, hue='val_accs_idx', size='val_accs_idx', legend='full')
    legend = ax.get_legend()
    legend.set_title('quantile')
    idxs = list(sorted(pca_features['val_accs'].unique()))
    for t in legend.texts:
        t.set_text(idxs[int(t._text)])

    def adjust_outliers(col, func, dif=0.05):
        scores = scipy.stats.zscore(col)
        col = [c for c, s in zip(col, scores) if abs(s) < 4]
        func(min(col) - dif, max(col) + dif)

    adjust_outliers(pca_features[0], plt.xlim)
    adjust_outliers(pca_features[1], plt.ylim)
    plt.title(title)
    plt.savefig(save)


def log_to_csv(out, out_prefix, timestamp, config_args, res_df, imp_df, pca_data, pca_train_data):
    if not os.path.exists(out):
        os.mkdir(out)

    name_args = {k: config_args[k] for k in ['benchmark', 'dataset', 'train_size', 'use_all_proxies']}
    name_args['model'] = config_args['model'] if 'model' in config_args else 'pca'
    out_name = '_'.join(f"{k}-{v}" for k, v in name_args.items())
    out_name = os.path.join(out, f"{out_prefix}{out_name}-{timestamp}")
    os.mkdir(out_name)

    with open(os.path.join(out_name, 'args.json'), 'w') as f:
        json.dump(config_args, f)

    def log_csv(df, name):
        df.to_csv(os.path.join(out_name, f'{name}.csv'), index=False)

    if res_df is not None:
        log_csv(res_df, 'res')
    if imp_df is not None:
        log_csv(imp_df, 'imp')

    _, pca_df = pca_data
    _, pca_train_df = pca_train_data
    
    _plot_pca(pca_data, 'Full fit', os.path.join(out_name, 'full_pca.png'))
    _plot_pca(pca_train_data, 'Train data fit', os.path.join(out_name, 'train_pca.png'))

    log_csv(pca_df, 'pca')
    log_csv(pca_train_df, 'pca_train_fit')


def log_to_wandb(key, project_name, timestamp, config_args, res_df, imp_df):
    import wandb
    wandb.login(key=key)
    wandb.init(project=project_name, config=config_args, name=timestamp)

    def log_stats(df, name, pref=''):
        if df is None:
            return
        wandb.log({name: wandb.Table(dataframe=df)})
        wandb.log({f"{pref}{k}_mean": np.mean(df[k]) for k in df.columns})
        wandb.log({f"{pref}{k}_std": np.std(df[k]) for k in df.columns})

    log_stats(res_df, 'results')
    log_stats(imp_df, 'feature_importances', pref='featimp_')


@click.command()
@click.option('--out', default='.', help='Root output directory.')
@click.option('--out_prefix', default='', help='Subdirectory prefix, the rest of the name has arg values.')
@click.option('--benchmark', required=True, help="Possible values: nb101, nb201, tnb101, nb301.")
@click.option('--searchspace_path', default='../data')
@click.option('--dataset', default='cifar10')
@click.option('--cfg', default=None, help="Path to the config (examples are in zc_combine/configs/")
@click.option('--meta', default=None, help="Path to meta.json for unique nb201 nets.")
@click.option('--features', default=None, help="Optionally pass comma-separated list of feature function "
                                               "names to use from the config (default is use all of them).")
@click.option('--proxy', default=None, help='Comma separated list of proxies to add to the dataset.')
@click.option('--columns_json', default=None, help='Json list of columns to use (e.g. based on feat imps).')
@click.option('--use_all_proxies/--not_all_proxies', default=False,
              help='If True, use all available proxies in the dataset.')
@click.option('--use_features/--no_features', default=True,
              help='If True, use simple features in the dataset.')
@click.option('--use_flops_params/--no_flops_params', default=True,
              help='If True, add flops and params to dataset regardless of what other proxies are chosen.')
@click.option('--filter_zero_op/--no_zero_op', default=False, help="If True, additionally filter out nb201"
                                                                   " nets with zero-ops isomorphic to other nets.")
@click.option('--only_pca/--pca_and_model', default=False, help="If True, fit PCA only and not the model.")
@click.option('--pca_loadings/--pca_components', default=True, help="If True, save loadings and not eigenvalues.")
@click.option('--standardize/--no_standardize', default=False, help="If True, apply StandardScaler before PCA.")
@click.option('--n_components', default=2, help="Number of PCA components.")
@click.option('--n_evals', default=10, help="Number of models fitted and evaluated on the data (with random"
                                            " state seed + i).")
@click.option('--seed', default=42, help="Starting seed.")
@click.option('--data_seed', default=42, help="Data split seed.")
@click.option('--train_size', default=100, help="Number of train architectures sampled.")
@click.option('--model', default='rf', help="Model to use (rf, xgb, xgb_tuned).")
@click.option('--wandb_key', default=None, help="If provided, data is logged to wandb instead.")
@click.option('--wandb_project', default='simple_features', help='Wandb project name (used only if '
                                                                 '--wandb_key is provided).')
def main(out, out_prefix, benchmark, searchspace_path, dataset, cfg, meta, features, proxy, columns_json,
         use_all_proxies, use_features, use_flops_params, filter_zero_op, only_pca, pca_loadings, standardize,
         n_components, n_evals, seed, data_seed, train_size, model, wandb_key, wandb_project):
    assert not (only_pca and wandb_key is not None), "Pca can be run only with local logging"

    # construct args for directory/wandb names
    cfg_args = {'benchmark': benchmark, 'dataset': dataset, 'n_evals': n_evals, 'seed': seed, 'data_seed': data_seed,
                'train_size': train_size, 'use_all_proxies': use_all_proxies,
                'use_features': use_features, 'proxy': None, 'features': None, 'standardize': standardize,
                'pca_loadings': pca_loadings}
    if not only_pca:
        cfg_args['model'] = model
    if not use_all_proxies:
        cfg_args['proxy'] = proxy
        cfg_args['use_flops_params'] = use_flops_params
    if use_features:
        cfg_args['features'] = features
    if columns_json is not None:
        cfg_args.update(parse_columns_filename(columns_json))

    # load meta.json to filter unique nets from nb201 and tnb101
    if meta is not None:
        with open(meta, 'r') as f:
            meta = json.load(f)

    data = load_bench_data(searchspace_path, benchmark, dataset, filter_nets=meta, zero_op_filtering=filter_zero_op)
    nets = get_net_data(data, benchmark)

    # To convert data['net'] to str: (4, 0, 3, 1, 4, 3)

    # from naslib.search_spaces.nasbench201.conversions import convert_op_indices_to_str
    # from zc_combine.fixes.operations import parse_ops_nb201
    # convert_op_indices_to_str(parse_ops_nb201(data.iloc[0]))
    # Out[5]: '|avg_pool_3x3~0|+|skip_connect~0|none~1|+|nor_conv_1x1~0|avg_pool_3x3~1|nor_conv_1x1~2|'

    if cfg is not None:
        with open(cfg, 'r') as f:
            cfg = json.load(f)

    features = features if features is None else features.split(',')
    proxy = proxy.split(',') if proxy is not None else []

    dataset, y = get_dataset(data, nets, benchmark, cfg=cfg, features=features, proxy_cols=proxy,
                             use_features=use_features,
                             use_all_proxies=use_all_proxies,
                             use_flops_params=use_flops_params)

    # select subset of columns based on previously saved data
    if columns_json is not None:
        with open(columns_json, 'r') as f:
            cols = json.load(f)
            cols = set(cols)
        dataset = dataset[cols]

    # train test split, access splits - res['train_X'], res['test_y'],...
    data_splits = get_data_splits(dataset, y, random_state=data_seed, train_size=train_size)

    result_df, importances_df = None, None
    if not only_pca:
        # fit model n times with different seeds
        model_cls = predictor_cls[model]
        fitted_models, res = eval_model(model_cls, data_splits, n_times=n_evals, random_state=seed)
        result_df = pd.DataFrame(res)

        importances_df = []
        for model, s in zip(fitted_models, res['seed']):
            imps = {c: i for c, i in zip(data_splits['train_X'].columns, model.feature_importances_)}
            importances_df.append({'seed': s, **imps})
        importances_df = pd.DataFrame(importances_df)

    timestamp = datetime.fromtimestamp(time.time()).strftime("%d-%m-%Y-%H-%M-%S")
    if wandb_key is not None:
        log_to_wandb(wandb_key, wandb_project, timestamp, cfg_args, result_df, importances_df)
    else:
        # do pca only for local runs (for feature selection)
        pca_data = do_pca(dataset, dataset, y, n_components, compute_loadings=pca_loadings, standardize=standardize)
        pca_train_data = do_pca(data_splits["train_X"], dataset, y, n_components, compute_loadings=pca_loadings, standardize=standardize)
        log_to_csv(out, out_prefix, timestamp, cfg_args, result_df, importances_df, pca_data, pca_train_data)


if __name__ == "__main__":
    main()
