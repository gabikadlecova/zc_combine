import argparse
import os.path
import json
import numpy as np
import pandas as pd
import wandb

from args_utils import parser_add_dataset_defaults, parse_and_read_args, log_dataset_args, parser_add_flag
from utils import load_feature_proxy_dataset, get_data_splits, eval_model, get_timestamp, create_cache_filename
from zc_combine.predictors import predictor_cls


def get_local_out_name(out, out_prefix, timestamp, config_args):
    if not os.path.exists(out):
        os.mkdir(out)

    name_args = {k: config_args[k] for k in ['benchmark', 'dataset', 'train_size', 'use_all_proxies']}
    name_args['model'] = config_args['model'] if 'model' in config_args else 'pca'
    out_name = '_'.join(f"{k}-{v}" for k, v in name_args.items())
    out_prefix = out_prefix if out_prefix is not None else ''
    out_prefix = f"{out_prefix}-" if len(out_prefix) else ''
    out_name = os.path.join(out, f"{out_prefix}{out_name}-{timestamp}")
    os.mkdir(out_name)

    # log args
    with open(os.path.join(out_name, 'args.json'), 'w') as f:
        json.dump(config_args, f)

    return out_name


def log_to_csv(out_name, config_args, res_df, imp_df):
    def log_csv(df, name):
        df.to_csv(os.path.join(out_name, f'{name}_{config_args["data_seed"]}.csv'), index=False)

    if res_df is not None:
        log_csv(res_df, 'res')
    if imp_df is not None:
        log_csv(imp_df, 'imp')


def log_to_wandb(data_seed, res_df, imp_df):
    def log_stats(df, name):
        if df is None:
            return
        # multiple fits per model
        if len(df) > 1:
            wandb.log({name: wandb.Table(dataframe=df)}, step=data_seed)
            wandb.log({f"{k}_mean": np.mean(df[k]) for k in df.columns}, step=data_seed)
            wandb.log({f"{k}_std": np.std(df[k]) for k in df.columns}, step=data_seed)
        else:
            wandb.log({k: df[k].iloc[0] for k in df.columns}, step=data_seed)

    log_stats(res_df, 'results')
    if imp_df is not None:
        log_stats(imp_df, 'feature_importances')


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
                                            use_onehot=args['use_onehot'],
                                            zero_unreachable=args['zero_unreachables'],
                                            keep_uniques=args['keep_uniques'],
                                            target_csv=args['target_csv_'],
                                            target_key=args['target_key'],
                                            cache_path=cache_path,
                                            version_key=args['version_key'])

    # select subset of columns based on previously saved data
    if args['columns_json_'] is not None:
        with open(args['columns_json_'], 'r') as f:
            cols = json.load(f)
            cols = set(cols)
        dataset = dataset[cols]

    # set up logging
    out_dir = None  # none for wandb cases
    if args['wandb_key_'] is not None:
        wandb.login(key=args['wandb_key_'])
        if args['restart_id_'] is None:
            wandb.init(project=args['wandb_project_'], config=cfg_args, name=get_timestamp())
        else:
            wandb.init(project=args['wandb_project_'], config=cfg_args, id=args['restart_id_'], resume='must')
    else:
        # set up out dir or locate the old one
        timestamp = get_timestamp() if args['restart_id_'] is None else args['restart_id_']
        out_dir = get_local_out_name(args['out_'], args['out_prefix'], timestamp, cfg_args)

    # run for multiple train set samples, test on the remainder
    for i_seed in range(args['n_train_samples']):
        # data seed for this run - skip if restarting
        curr_data_seed = args['data_seed'] + i_seed
        cfg_args['data_seed'] = curr_data_seed
        if args['restart_id_'] is not None and args['restart_data_seed_'] > curr_data_seed:
            continue

        # train test split, access splits - res['train_X'], res['test_y'],...
        data_splits = get_data_splits(dataset, y, random_state=curr_data_seed, train_size=args['train_size'])

        # fit model n_evals times with different seeds
        model_cls = predictor_cls[args['model']]
        fitted_models, res = eval_model(model_cls, data_splits, n_times=args['n_evals'], random_state=args['seed'])
        result_df = pd.DataFrame(res)

        # log feature importances
        importances_df = None
        if args['log_featimp_']:
            importances_df = []
            for model, s in zip(fitted_models, res['seed']):
                imps = {c: i for c, i in zip(data_splits['train_X'].columns, model.feature_importances_)}
                importances_df.append({'seed': s, **imps})
            importances_df = pd.DataFrame(importances_df)

        # log either to wandb or to local dataframes
        if args['wandb_key_'] is not None:
            log_to_wandb(curr_data_seed, result_df, importances_df)
        else:
            log_to_csv(out_dir, cfg_args, result_df, importances_df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Computes PCA from simple features, saves loadings/coefficients, plots it."
    )

    parser_add_dataset_defaults(parser)

    parser.add_argument('--out_', type=str, help="Directory for output saving (subdir with timestamp will"
                                                 " be created).")
    parser.add_argument('--out_prefix', default=None, type=str, help="Prefix of the subdirectory.")
    parser.add_argument('--columns_json_', default=None, type=str,
                        help="Json list of columns to use (e.g. based on feat imps).")
    parser.add_argument('--wandb_key_', default=None, type=str, help="If provided, data is logged to wandb instead.")
    parser.add_argument('--wandb_project_', default=None, type=str, help="Wandb project name (used only if "
                                                                         "--wandb_key_ is provided).")
    parser.add_argument('--n_evals', default=1, type=int, help="Number of models fitted and evaluated on the data "
                                                               "(with random state seed + i).")
    parser.add_argument('--seed', default=42, type=int, help="Starting model seed.")
    parser.add_argument('--model', default='rf', type=str, help="Model to use (rf, xgb, xgb_tuned).")
    parser.add_argument('--n_train_samples', default=1, type=int, help="Number of times a train set is sampled from the"
                                                                       " dataset (with random state data_seed + i)")
    parser.add_argument('--restart_id_', default=None, type=str, help="Id of the run to restart.")
    parser.add_argument('--restart_data_seed_', default=42, type=int,
                        help="Seed (absolute) for restart, will skip seeds "
                             "smaller than this value. Runs for all seeds between [restart_data_seed, data_seed + n_train_samples)")
    parser_add_flag(parser, 'log_featimp_', 'no_log_featimp', False, help_neg="If True, log feature importances.")

    args = parse_and_read_args(parser)

    train_and_eval(args)
