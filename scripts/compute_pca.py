import argparse
import os.path

import json
import numpy as np
import pandas as pd

import scipy.stats
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from utils import get_data_splits, load_feature_proxy_dataset, get_timestamp, create_cache_filename
from args_utils import parser_add_dataset_defaults, parser_add_flag, parse_and_read_args, log_dataset_args


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
    plot_dd['val_accs'] = pd.qcut(transform_y.reset_index(drop=True), 4)
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


def log_to_csv(out, out_prefix, timestamp, config_args, pca_data, pca_train_data, plot):
    if not os.path.exists(out):
        os.mkdir(out)

    name_args = {k: config_args[k] for k in ['benchmark', 'dataset', 'train_size']}
    out_prefix = '' if out_prefix is None else f"{out_prefix}-"
    out_name = '_'.join(f"{k}-{v}" for k, v in name_args.items())
    out_name = os.path.join(out, f"{out_prefix}pca-{out_name}-{timestamp}")
    os.mkdir(out_name)

    with open(os.path.join(out_name, 'args.json'), 'w') as f:
        json.dump(config_args, f)

    def log_csv(df, name):
        df.to_csv(os.path.join(out_name, f'{name}.csv'), index=False)

    _, pca_df = pca_data
    _, pca_train_df = pca_train_data

    if plot:
        _plot_pca(pca_data, 'Full fit', os.path.join(out_name, 'full_pca.png'))
        _plot_pca(pca_train_data, 'Train data fit', os.path.join(out_name, 'train_pca.png'))

    log_csv(pca_df, 'pca')
    log_csv(pca_train_df, 'pca_train')
    print(out_name)


def run_pca(args):
    cfg_args = log_dataset_args(args)

    cache_path = None
    if args['use_features'] and args['cache_dir_'] is not None:
        cache_path = create_cache_filename(args['cache_dir_'], args['cfg'], args['features'], args['version_key'])

    dataset, y = load_feature_proxy_dataset(args['searchspace_path_'], args['benchmark'], args['dataset'],
                                            cfg=args['cfg'], features=args['features'], proxy=args['proxy'],
                                            meta=args['meta'], use_features=args['use_features'],
                                            use_all_proxies=args['use_all_proxies'],
                                            use_flops_params=args['use_flops_params'],
                                            zero_unreachable=args['zero_unreachables'],
                                            keep_uniques=args['keep_uniques'],
                                            cache_path=cache_path,
                                            version_key=args['version_key'])

    # train test split, access splits - res['train_X'], res['test_y'],...
    data_splits = get_data_splits(dataset, y, random_state=args['data_seed'], train_size=args['train_size'])

    n_components, loadings, standardize = args['n_components'], args['pca_loadings'], args['standardize']
    pca_data = do_pca(dataset, dataset, y, n_components, compute_loadings=loadings, standardize=standardize)
    pca_train_data = do_pca(data_splits["train_X"], dataset, y, n_components, compute_loadings=loadings,
                            standardize=standardize)
    log_to_csv(args['out_'], args['out_prefix'], get_timestamp(), cfg_args, pca_data, pca_train_data, args['plot_'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Computes PCA from simple features, saves loadings/coefficients, plots it."
    )

    parser_add_dataset_defaults(parser)

    parser.add_argument('--out_', type=str, help="Directory for output saving (subdir with timestamp will"
                        " be created).")
    parser.add_argument('--out_prefix', default=None, type=str, help="Prefix of the subdirectory.")
    parser.add_argument('--n_components', default=2, type=int, help="Number of PCA components computed.")
    parser_add_flag(parser, 'plot_', 'no_plot_', False, help_pos="Plot PCA and save it to out dir.")
    parser_add_flag(parser, 'standardize', 'no_standardize', False,
                    help_pos="Standardize data before doing PCA.")
    parser_add_flag(parser, 'pca_loadings', 'pca_coef', False,
                    help_pos="Return loadings: pca.components_ * np.sqrt(pca.explained_variance_)",
                    help_neg="Return only pca.components_")

    args = parse_and_read_args(parser)

    run_pca(args)
