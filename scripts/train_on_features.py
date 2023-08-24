import os.path

import click
import json

import pandas as pd

from utils import load_bench_data, get_net_data, get_dataset, get_data_splits, eval_model
from zc_combine.predictors import predictor_cls


@click.command()
@click.option('--out', required=True)
@click.option('--benchmark', required=True, help="Possible values: nb101, nb201, tnb101, nb301.")
@click.option('--searchspace_path', default='../data')
@click.option('--dataset', default='cifar10')
@click.option('--cfg', default=None, help="Path to the config (examples are in zc_combine/configs/")
@click.option('--meta', default=None, help="Path to meta.json for unique nb201 nets.")
@click.option('--features', default=None, help="Optionally pass comma-separated list of feature function "
                                               "names to use from the config (default is use all of them).")
@click.option('--proxy', default=None, help='Comma separated list of proxies to add to the dataset.')
@click.option('--use_all_proxies/--not_all_proxies', default=False,
              help='If True, use all available proxies in the dataset.')
@click.option('--use_features/--no_features', default=True,
              help='If True, use simple features in the dataset.')
@click.option('--use_flops_params/--no_flops_params', default=True,
              help='If True, add flops and params to dataset regardless of what other proxies are chosen.')
@click.option('--n_evals', default=10, help="Number of models fitted and evaluated on the data (with random"
                                            " state seed + i).")
@click.option('--seed', default=42, help="Starting seed.")
@click.option('--data_seed', default=42, help="Data split seed.")
@click.option('--train_size', default=100, help="Number of train architectures sampled.")
@click.option('--model', default='rf', help="Model to use (rf, xgb, xgb_tuned).")
def main(out, benchmark, searchspace_path, dataset, cfg, meta, features, proxy, use_all_proxies, use_features,
         use_flops_params, n_evals, seed, data_seed, train_size, model):

    # load meta.json to filter unique nets from nb201 and tnb101
    if meta is not None:
        with open(meta, 'r') as f:
            meta = json.load(f)

    data = load_bench_data(searchspace_path, benchmark, dataset, filter_nets=meta)
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

    # train test split, access splits - res['train_X'], res['test_y'],...
    data_splits = get_data_splits(dataset, y, random_state=data_seed, train_size=train_size)

    # fit model n times with different seeds
    model_cls = predictor_cls[model]
    fitted_models, res = eval_model(model_cls, data_splits, n_times=n_evals, random_state=seed)

    importances_df = []
    for model, s in zip(fitted_models, res['seed']):
        imps = {c: i for c, i in zip(data_splits['train_X'].columns, model.feature_importances_)}
        importances_df.append({'seed': s, **imps})
    importances_df = pd.DataFrame(importances_df)

    # TODO wandb
    # TODO better output path
    result_df = pd.DataFrame(res)
    if not os.path.exists(out):
        os.mkdir(out)

    result_df.to_csv(os.path.join(out, 'res.csv'))
    importances_df.to_csv(os.path.join(out, 'imp.csv'))
    print(res)
    # TODO PCA


if __name__ == "__main__":
    main()
