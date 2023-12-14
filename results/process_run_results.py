import warnings

import click
import pandas as pd
import wandb

from tqdm import tqdm

DEFAULT_CFG_ARGS = ['cfg', 'dataset', 'data_seed', 'proxy', 'train_size', 'use_all_proxies', 'use_features',
                    'use_flops_params', 'use_onehot', 'use_path_encoding']
DEFAULT_HISTORY = ['tau', 'corr', 'fit_time', 'test_time', '_step']


@click.command()
@click.option('--key', help="Wandb API key")
@click.option('--project', help="Wandb project name with runs to analyse", required=True)
@click.option('--timeout', type=int, help="Timeout for runs (wandb suggests a value after timeouts). "
                                          "Suggested values: 19, 29, 39.")
@click.option('--cfg_args', default=None)
@click.option('--metric_keys', default=None)
@click.option('--out_path', required=True)
@click.option('--keep_all_history/--aggregate_history', default=False,
              help="If True, duplicate config rows and keep all metric values from different seeds. Otherwise keep "
                   "only mean and std of a metric (in a single row).")
def main(key, project, timeout, cfg_args, metric_keys, out_path, keep_all_history):
    cfg_args = DEFAULT_CFG_ARGS if cfg_args is None else cfg_args.split(',')
    metric_keys = DEFAULT_HISTORY if metric_keys is None else metric_keys.split(',')

    wandb.login(key=key)

    api = wandb.Api(timeout=timeout)
    runs = api.runs(project)  # example project: "USER_NAME" + "/" + project

    results_df = []
    for run in tqdm(runs):
        row = {}
        for ca in cfg_args:
            if ca not in run.config:
                warnings.warn(f"Key {ca} is not found in config.")
                row[ca] = None
                continue

            row[ca] = run.config[ca]

        history_df = run.history()[metric_keys]
        if '_step' in metric_keys:
            history_df.rename(columns={'_step': 'data_seed'}, inplace=True)

        # add row(s) to final results df
        if keep_all_history:
            # duplicate rows with the same config values (metrics are from history_df)
            rows = [{**row, **{k: history_df.iloc[i][k] for k in history_df.columns}} for i in range(len(history_df))]
            results_df.extend(rows)
        else:
            for col in history_df.columns:
                row[f"{col}_mean"] = history_df[col].mean()
                row[f"{col}_std"] = history_df[col].std()
            results_df.append(row)

    results_df = pd.DataFrame(results_df)
    results_df.to_csv(out_path)


if __name__ == "__main__":
    main()
