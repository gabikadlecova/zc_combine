import pickle
import warnings

import click
import pandas as pd
import wandb

from tqdm import tqdm

DEFAULT_CFG_ARGS = ['cfg', 'dataset', 'data_seed', 'proxy', 'train_size', 'use_all_proxies', 'use_features',
                    'use_flops_params', 'use_onehot', 'use_path_encoding', 'features']
DEFAULT_HISTORY = ['tau', 'corr', 'fit_time', 'test_time', '_step']


@click.command()
@click.option('--key', help="Wandb API key")
@click.option('--project', help="Wandb project name with runs to analyse", required=True)
@click.option('--timeout', type=int, help="Timeout for runs (wandb suggests a value after timeouts). "
                                          "Suggested values: 19, 29, 39.")
@click.option('--cfg_args', default=None)
@click.option('--metric_keys', default=None)
@click.option('--out_path', required=True)
@click.option('--nan_check_path', default=None)
@click.option('--page_size', default=50, help="Increase only if there is a duplication bug in wandb.")
@click.option('--history_len', default=50, help="Required length of history (for each run).")
@click.option('--keep_all_history/--aggregate_history', default=False,
              help="If True, duplicate config rows and keep all metric values from different seeds. Otherwise keep "
                   "only mean and std of a metric (in a single row).")
def main(key, project, timeout, cfg_args, metric_keys, out_path, nan_check_path, page_size, history_len,
         keep_all_history):
    cfg_args = DEFAULT_CFG_ARGS if cfg_args is None else cfg_args.split(',')
    metric_keys = DEFAULT_HISTORY if metric_keys is None else metric_keys.split(',')

    wandb.login(key=key)

    api = wandb.Api(timeout=timeout)
    runs = api.runs(project, per_page=page_size)  # example project: "USER_NAME" + "/" + project

    # check if all seeds ran for all runs and if no nans are present
    incomplete_runs = []

    visited = {}

    results_df = []
    for run in tqdm(runs):
        if run.id in visited:
            raise ValueError(f"Run {run.id} is duplicated. Try to increase page size to the number of runs, "
                             f"or add and remove a run in the project.")

        visited.add(run.id)

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

        # check if all seeds ran without errors
        short_history = len(history_df) < history_len
        if short_history:
            warnings.warn(f"Run {run.id} has only {len(history_df)} data_seed entries instead of {history_len}.")

        has_nans = history_df.isna().any().any()
        if has_nans:
            warnings.warn(f"Run {run.id} has nans in results.")

        if short_history or has_nans:
            incomplete_runs.append(run.id)

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

    if nan_check_path is not None:
        with open(nan_check_path, 'wb') as f:
            pickle.dump(incomplete_runs, f)


if __name__ == "__main__":
    main()
