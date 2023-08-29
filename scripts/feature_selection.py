import json
import os.path

import click
import numpy as np
import pandas as pd


def get_cols_list(imps, cols, n_cols):
    imps = np.abs(imps)
    cidx = np.argsort(imps)
    return cols[cidx][::-1][:n_cols].tolist()


@click.command()
@click.option('--imp_path', required=True)
@click.option('--out_prefix', default='', help="Output file name prefix")
@click.option('--n_features', default=20)
@click.option('--mode', default='mean', help='One of (mean, row, idx). Mean means aggregating scores over seeds, row '
                                             'is selecting n features // n rows for every row, idx is selecting '
                                             'features for idx-th row.')
@click.option('--idx', default=0, help="If mode == idx, id of the row for feature selection.")
def main(imp_path, out_prefix,  n_features, mode, idx):
    imps = pd.read_csv(imp_path)

    if mode == 'mean':
        imps = imps.mean(axis=0)
        res_cols = get_cols_list(imps.to_numpy(), imps.index, n_features)
    elif mode == 'row':
        names = [get_cols_list(imps.loc[i].to_numpy(), imps.columns, n_features) for i in imps.index]
        res_cols = [c for cols in names for c in cols]

    elif mode == 'idx':
        imps = imps.iloc[idx]
        res_cols = get_cols_list(imps.to_numpy(), imps.index, n_features)

    else:
        raise ValueError("Invalid mode")

    res_cols = list(set(res_cols))

    out_prefix = f"{out_prefix}-" if len(out_prefix) else ''
    out_path, ext = os.path.splitext(imp_path)
    out_path = f"{out_path}-{out_prefix}{n_features}-{mode}{'' if mode != 'idx' else f'-{idx}'}.json"
    with open(out_path, 'w') as f:
        json.dump(res_cols, f)

    print(out_path)


if __name__ == "__main__":
    main()
