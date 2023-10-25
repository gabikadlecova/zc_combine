import click
import glob
import itertools
import json
import os
import pandas as pd
import pdb

def parse_const(val):
    # range, e.g. 1-10 (upper exclusive)
    if '-' in val:
        splitted = val.split('-')
        if len(splitted) == 2 and str.isnumeric(splitted[0]) and str.isnumeric(splitted[1]):
            val = [i for i in range(int(splitted[0]), int(splitted[1]))]

    return val


def read_args(arg_dict, args_name):
    res = []
    for k, vals in arg_dict.items():
        if not k.startswith('_'):
            vals = vals if isinstance(vals, list) else parse_const(vals)

            # constant arg
            if not isinstance(vals, list):
                res.append([f"--{k} {vals}"])
            # arg with variants
            else:
                res.append([f"--{k} {v}" for v in vals])
        elif not k.startswith('_raw'):
            raise ValueError(f"Invalid argname: {k}")
        else:
            res.append(vals)

    # append column name to all args
    res = [[(args_name, r) for r in subres] for subres in res]
    return res


@click.command()
@click.option('--dir_path', required=True, help="Directory with json config args.")
def main(dir_path):
    all_args = []

    for argfile in glob.glob(os.path.join(dir_path, "*.json")):
        colname = os.path.splitext(os.path.basename(argfile))[0]

        with open(argfile, 'r') as f:
            args_list = read_args(json.load(f), colname)
            all_args.extend(args_list)

    # create database of settings
    df = []
    for arg_setting in itertools.product(*all_args):
        row = {}
        for (name, val) in arg_setting:
            arglist = row.setdefault(name, [])
            arglist.append(val)
        row = {k: ' '.join(r) for k, r in row.items()}
        df.append(row)

    if not len(all_args):
        raise ValueError(f"The directory did not contain any jsons: {dir_path}")

    df = pd.DataFrame(df)
    df.to_csv(os.path.join(dir_path, 'experiment_settings.csv'))


if __name__ == "__main__":
    main()
