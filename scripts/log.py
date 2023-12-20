import json
import os
import time
from datetime import datetime

import numpy as np
import wandb


def get_local_out_name(out, out_prefix, timestamp, config_args, conf_names=None):
    if not os.path.exists(out):
        os.mkdir(out)

    def_names = ['benchmark', 'dataset', 'train_size', 'use_all_proxies', 'model']
    conf_names = conf_names if conf_names is not None else def_names

    name_args = {k: config_args[k] for k in conf_names}
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


def get_timestamp():
    return datetime.fromtimestamp(time.time()).strftime("%d-%m-%Y-%H-%M-%S-%f")


def set_up_logging(cfg_args, wandb_key=None, wandb_project=None, restart_id=None, out=None, out_prefix=None,
                   conf_names=None):
    # set up logging
    assert out is not None or wandb_key is not None, "Either out dir or wandb key must be set."

    out_dir = None  # none for wandb cases
    if wandb_key is not None:
        wandb.login(key=wandb_key)
        if restart_id is None:
            wandb.init(project=wandb_project, config=cfg_args, name=get_timestamp())
        else:
            wandb.init(project=wandb_project, config=cfg_args, id=restart_id, resume='must')
    else:
        # set up out dir or locate the old one
        timestamp = get_timestamp() if restart_id is None else restart_id
        out_dir = get_local_out_name(out, out_prefix, timestamp, cfg_args,
                                     conf_names=conf_names)
    return out_dir
