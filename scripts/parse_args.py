import json
import os.path

import click
import pandas as pd


def get_arg(k, v):
    if isinstance(v, bool):
        return f'--{k}' if v else ''
    return f'--{k} {v}'


@click.command()
@click.argument('path', required=True)
def main(path):
    assert os.path.exists(path)

    if os.path.isdir(path):
        res_df = []
        index = []
        for subdir in os.listdir(path):
            args_name = os.path.join(path, subdir, 'args.json')
            if not os.path.exists(args_name):
                continue

            with open(args_name, 'r') as f:
                args = json.load(f)

            res_df.append({k: v for k, v in args.items() if v is not None})
            index.append(subdir)

        res_df = pd.DataFrame(data=res_df, index=index)
        res_df.to_csv(os.path.join(path, 'dir_args.csv'))
    else:
        with open(path, 'r') as f:
            args = json.load(f)
        print(' '.join(get_arg(k, v) for k, v in args.items() if v is not None))


if __name__ == "__main__":
    main()
