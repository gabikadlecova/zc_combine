import json
import logging
import sys
import os
import warnings


def parse_and_read_args(parser):
    args = parser.parse_args()
    args = vars(args)
    arg_file = args['args_json_']

    cmd_args = {s[2:] for s in sys.argv if s.startswith('--')}

    # overwrite command line args with args from file
    if arg_file is not None:
        with open(arg_file, 'r') as f:
            arg_file = json.load(f)
            # load all args except those passed to the script on the command line
            for k, v in arg_file.items():
                if k not in cmd_args:
                    args[k] = v

    return args


def parser_add_flag(parser, flag_pos, flag_neg, default, flag_name=None, help_pos=None, help_neg=None):
    flag_name = flag_pos if flag_name is None else flag_name
    parser.add_argument(f'--{flag_pos}', dest=flag_name, action='store_true', help=help_pos)
    parser.add_argument(f'--{flag_neg}', dest=flag_name, action='store_false', help=help_neg)
    parser.set_defaults(**{flag_name: default})


def log_dataset_args(args):
    cfg_args = {k: v for k, v in args.items() if not k.endswith('_')}

    if args['use_all_proxies']:
        cfg_args['proxy'] = None
        cfg_args['use_flops_params'] = None

    if not args['use_features']:
        args['features'] = None

    return cfg_args


def parser_add_dataset_defaults(parser):
    parser.add_argument('--args_json_', default=None, help="Json file with args; will overwrite args passed on the command line.")
    parser.add_argument('--searchspace_path_', default='../data', help="Directory with json files of proxy scores "
                                                                      "(from NASLib).")
    parser.add_argument('--cache_dir_', default=None, help="Path to cache the feature datasets - filenames are composed "
                                                           "from cfg file name, features and version_key.")
    parser.add_argument('--version_key', default=None, help="Version key of the cached dataset.")
    parser.add_argument('--benchmark', default='nb201', help="Which NAS benchmark to use (e.g. nb201).")
    parser.add_argument('--dataset', default='cifar10', help="Which dataset from the benchmark to use (e.g. cifar10).")
    parser.add_argument('--data_seed', default=42, type=int, help="Seed for dataset splits.")
    parser.add_argument('--train_size', default=100, type=int, help="Train split size.")
    parser.add_argument('--proxy', default=None, type=str, help="Comma separated list of proxies to use.")
    parser.add_argument('--features', default=None, type=str, help="Comma separated list of features to use.")
    parser.add_argument('--cfg', default=None, type=str, help="Path to config file for proxy dataset creation. Example configs are `zc_combine/configs/*.json`.")
    parser.add_argument('--meta', default=None, type=str, help="Path to the json file with unique nets filtered out (for nb201 and tnb101).")
    parser.add_argument('--target_csv_', default=None, type=str, help="Optional dataframe with user defined targets.")
    parser.add_argument('--target_key', default='val_accs', type=str, help="Name of the target to be predicted.")
    parser_add_flag(parser, 'use_all_proxies', 'not_all_proxies', False,
                    help_pos="Use all available proxies.", help_neg="Use only selected proxies.")
    parser_add_flag(parser, 'use_features', 'no_features', True,
                    help_neg="Use only proxies and not features.")
    parser_add_flag(parser, 'use_flops_params', 'no_flops_params', True,
                    help_pos="Add flops, params regardless of other proxy settings")
    parser_add_flag(parser, 'zero_unreachables', 'no_zero_unreachables', True,
                    help_pos="Zero out unreachable ops, keep only the default (networks that are the same before and "
                             "after the zeroing).", help_neg="Keep all networks with unreachable ops.")
    parser_add_flag(parser, 'keep_uniques', 'all_networks', True,
                    help_pos="Keep only unique networks - remove isomorphisms.",
                    help_neg="Keep all networks (including isomorphic networks).")
