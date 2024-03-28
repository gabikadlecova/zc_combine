import argparse
import json
import pandas as pd
import pdb

from zc_combine.utils.args_utils import parser_add_dataset_defaults, parse_and_read_args, log_dataset_args, parser_add_flag
from zc_combine.utils.log import set_up_logging, log_to_wandb, log_to_csv
from zc_combine.utils.script_utils import load_feature_proxy_dataset, get_data_splits, eval_model, \
    create_cache_filename, get_wl_embedding, normalize_columns, shap_importances

from sklearn.ensemble import RandomForestRegressor

from zc_combine.predictors import predictor_cls


def train_and_eval(args):
    cfg_args = log_dataset_args(args)

    cache_path = None
    if args['use_features'] and args['cache_dir_'] is not None:
        cache_path = create_cache_filename(args['cache_dir_'], args['cfg'], args['features'], args['version_key'],
                                           args['compute_all_'])

    net_data, dataset, y = load_feature_proxy_dataset(args['searchspace_path_'], args['benchmark'], args['dataset'],
                                                      cfg=args['cfg'], features=args['features'], proxy=args['proxy'],
                                                      meta=args['meta'], use_features=args['use_features'],
                                                      use_all_proxies=args['use_all_proxies'],
                                                      use_flops_params=args['use_flops_params'],
                                                      use_onehot=args['use_onehot'],
                                                      use_embedding=args['use_embedding'], 
                                                      use_path_encoding=args['use_path_encoding'],
                                                      zero_unreachable=args['zero_unreachables'],
                                                      keep_uniques=args['keep_uniques'],
                                                      target_csv=args['target_csv_'],
                                                      target_key=args['target_key'],
                                                      cache_path=cache_path,
                                                      version_key=args['version_key'],
                                                      compute_all=args['compute_all_'],
                                                      multi_objective=args["multi_objective"],
                                                      replace_bad=args['replace_bad'])

    # select subset of columns based on previously saved data
    if args['columns_json_'] is not None:
        with open(args['columns_json_'], 'r') as f:
            cols = json.load(f)
            cols = set(cols)
        dataset = dataset[cols]

    # set up logging - init wandb or local dir. Out_dir is None if wandb is used.
    out_dir = set_up_logging(cfg_args, wandb_key=args['wandb_key_'], wandb_project=args['wandb_project_'],
                             restart_id=args['restart_id_'], out=args['out_'],out_prefix=args['out_prefix'])

    # run for multiple train set samples, test on the remainder
    for i_seed in range(args['n_train_samples']):
        # data seed for this run - skip if restarting
        curr_data_seed = args['data_seed'] + i_seed
        cfg_args['data_seed'] = curr_data_seed
        if args['restart_id_'] is not None and args['restart_data_seed_'] > curr_data_seed:
            continue

        # train test split, access splits - res['train_X'], res['test_y'],...
        data_splits = get_data_splits(dataset, y, random_state=curr_data_seed, train_size=args['train_size'])
        if args['normalize_proxies']:
            data_splits = normalize_columns(data_splits, net_data['proxy_columns'])

        ##### use train data to fit WL Kernel here
        if args['use_wl_embedding']:
            data_splits = get_wl_embedding(data_splits, args['benchmark'])

        # remove net here again, which was included for WL initialization...
        if 'net' in data_splits['train_X'].columns:
            data_splits['train_X'].drop(columns='net', inplace=True)
        if 'net' in data_splits['test_X'].columns:
            data_splits['test_X'].drop(columns='net', inplace=True)

        # fit model n_evals times with different seeds
        model_cls = predictor_cls[args['model']]
        fitted_models, res = eval_model(model_cls, data_splits, n_times=args['n_evals'], random_state=args['seed'])
        result_df = pd.DataFrame(res)

        # log feature importances
        importances_df = None
        if args['log_featimp_']:
            importances_df = []
            for model, s in zip(fitted_models, res['seed']):
                if isinstance(model, RandomForestRegressor):
                    imps = {c: i for c, i in zip(data_splits['train_X'].columns, shap_importances(model, data_splits['test_X']))}
                else:
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
        description="Run fit-eval of features/zero-cost proxies predictor."
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
    parser_add_flag(parser, 'normalize_proxies', 'no_normalize_proxies', False,
                    help_pos="If True, normalize proxy columns in data using the train set.")

    args = parse_and_read_args(parser)

    train_and_eval(args)
