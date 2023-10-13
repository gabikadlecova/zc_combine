import argparse
import os.path
import json
import numpy as np
import pandas as pd

from args_utils import parser_add_dataset_defaults, parse_and_read_args, log_dataset_args
from utils import load_feature_proxy_dataset, get_data_splits, eval_model, get_timestamp, create_cache_filename
from zc_combine.predictors import predictor_cls


def log_to_csv(out, out_prefix, timestamp, config_args, res_df, imp_df):
    if not os.path.exists(out):
        os.mkdir(out)

    name_args = {k: config_args[k] for k in ['benchmark', 'dataset', 'train_size', 'use_all_proxies']}
    name_args['model'] = config_args['model'] if 'model' in config_args else 'pca'
    out_name = '_'.join(f"{k}-{v}" for k, v in name_args.items())
    out_prefix = out_prefix if out_prefix is not None else ''
    out_prefix = f"{out_prefix}-" if len(out_prefix) else ''
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


def train_and_eval(args):
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

    # select subset of columns based on previously saved data
    if args['columns_json_'] is not None:
        with open(args['columns_json_'], 'r') as f:
            cols = json.load(f)
            cols = set(cols)
        dataset = dataset[cols]

    # train test split, access splits - res['train_X'], res['test_y'],...
    data_splits = get_data_splits(dataset, y, random_state=args['data_seed'], train_size=args['train_size'])

    # To convert data['net'] to str: (4, 0, 3, 1, 4, 3)

    # from naslib.search_spaces.nasbench201.conversions import convert_op_indices_to_str
    # from zc_combine.fixes.operations import parse_ops_nb201
    # network_string = (4, 0, 3, 1, 4, 3)
    # convert_op_indices_to_str(parse_ops_nb201(network_string))
    # Out[5]: '|avg_pool_3x3~0|+|skip_connect~0|none~1|+|nor_conv_1x1~0|avg_pool_3x3~1|nor_conv_1x1~2|'

    # fit model n times with different seeds
    model_cls = predictor_cls[args['model']]
    fitted_models, res = eval_model(model_cls, data_splits, n_times=args['n_evals'], random_state=args['seed'])
    result_df = pd.DataFrame(res)

    importances_df = []
    for model, s in zip(fitted_models, res['seed']):
        imps = {c: i for c, i in zip(data_splits['train_X'].columns, model.feature_importances_)}
        importances_df.append({'seed': s, **imps})
    importances_df = pd.DataFrame(importances_df)

    timestamp = get_timestamp()
    if args['wandb_key_'] is not None:
        log_to_wandb(args['wandb_key_'], args['wandb_project_'], timestamp, cfg_args, result_df, importances_df)
    else:
        log_to_csv(args['out_'], args['out_prefix'], timestamp, cfg_args, result_df, importances_df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Computes PCA from simple features, saves loadings/coefficients, plots it."
    )

    parser_add_dataset_defaults(parser)

    parser.add_argument('--out_', type=str, help="Directory for output saving (subdir with timestamp will"
                        " be created).")
    parser.add_argument('--out_prefix', default=None, type=str, help="Prefix of the subdirectory.")
    parser.add_argument('--columns_json_', default=None, type=str, help="Json list of columns to use (e.g. based on feat imps).")
    parser.add_argument('--wandb_key_', default=None, type=str, help="If provided, data is logged to wandb instead.")
    parser.add_argument('--wandb_project_', default=None, type=str, help="Wandb project name (used only if "
                                                                         "--wandb_key_ is provided).")
    parser.add_argument('--n_evals', default=10, type=int, help="Number of models fitted and evaluated on the data "
                                                                "(with random state seed + i).")
    parser.add_argument('--seed', default=42, type=int, help="Starting seed.")
    parser.add_argument('--model', default='rf', type=str, help="Model to use (rf, xgb, xgb_tuned).")

    args = parse_and_read_args(parser)

    train_and_eval(args)
