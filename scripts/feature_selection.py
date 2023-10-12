import json
import os.path

import click
import numpy as np
import pandas as pd


def get_cols_list(imps, cols, n_cols=None, threshold=None):
    if n_cols is not None:
        imps = np.abs(imps)
        cidx = np.argsort(imps)
        return cols[cidx][::-1][:n_cols].tolist()

    return cols[np.abs(imps) >= threshold].tolist()


@click.command()
@click.option('--imp_path', required=True)
@click.option('--out_prefix', default='', help="Output file name prefix")
@click.option('--n_features', default=None, type=int)
@click.option('--threshold', default=None, type=float)
@click.option('--mode', default='mean', help='One of (mean, row, idx, norm). Mean means aggregating scores over seeds,'
                                             'row is selecting n features // n rows for every row, idx is selecting '
                                             'features for idx-th row, norm is l2 norm of rows.')
@click.option('--idx', default=0, help="If mode == idx, id of the row for feature selection.")
def main(imp_path, out_prefix, n_features, threshold, mode, idx):
    assert n_features is not None or threshold is not None, "Must provide either n_features or threshold for feature selection."

    imps = pd.read_csv(imp_path)

    if mode == 'mean':
        imps = imps.mean(axis=0)
        res_cols = get_cols_list(imps.to_numpy(), imps.index, n_cols=n_features, threshold=threshold)
    elif mode == 'row':
        names = [get_cols_list(imps.loc[i].to_numpy(), imps.columns, n_cols=n_features, threshold=threshold) for i in imps.index]
        res_cols = [c for cols in names for c in cols]

    elif mode == 'idx':
        imps = imps.iloc[idx]
        res_cols = get_cols_list(imps.to_numpy(), imps.index, n_cols=n_features, threshold=threshold)
    elif mode == 'norm':
        imps = sum(imps.iloc[i] ** 2 for i in range(len(imps)))
        res_cols = get_cols_list(imps.to_numpy(), imps.index, n_cols=n_features, threshold=threshold)
    else:
        raise ValueError("Invalid mode")

    res_cols = list(set(res_cols))

    out_prefix = f"{out_prefix}-" if len(out_prefix) else ''
    out_path, ext = os.path.splitext(imp_path)
    out_path = f"{out_path}-{out_prefix}{f'f_{n_features}' if n_features is not None else f't_{threshold}'}-{mode}{'' if mode != 'idx' else f'-{idx}'}.json"
    with open(out_path, 'w') as f:
        json.dump(res_cols, f)

    print(out_path)


if __name__ == "__main__":
    main()
