import click
import pickle
import pandas as pd
from zc_combine.fixes.operations import get_ops_edges_nb201


def convert_net(net, edge_indices, n_nodes=4):
    res = [1 for _ in edge_indices]  # default is zeros

    for i in range(1, n_nodes + 1):
        for j in range(i + 1, n_nodes + 1):
            idx = edge_indices[(i, j)]
            res[idx] = net[i - 1, j - 1]

    ta_idx_to_naslib = {0: 1, 1: 0, 2: 3, 3: 2, 4: 4}
    return str(tuple([ta_idx_to_naslib[i] for i in res]))


@click.command()
@click.argument('data_path')
@click.option('--out_csv', help="Output path to nets in tuple encoding.")
@click.option('--valids_csv', default=None, help="Dataframe with valid networks (no unreachable branches).")
@click.option('--out_valid_path', default=None, help="Save original nets but filtered.")
def main(data_path, out_csv, valids_csv, out_valid_path):
    _, edges = get_ops_edges_nb201()
    edge_indices = {e: i for i, e in enumerate(edges.keys())}

    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    valid_nets = []
    if valids_csv is not None:
        valids_csv = pd.read_csv(valids_csv)
        valid_nets = valids_csv['net'].tolist()

    out_valid_data = None if out_valid_path is None else []
    res_df = []
    for net in data:
        new_net = convert_net(net[0].T, edge_indices)
        if new_net not in valid_nets:
            continue

        if out_valid_data is not None:
            out_valid_data.append(net)

        acc = net[2]
        res_df.append({'net': new_net, 'val_accs': acc})

    res_df = pd.DataFrame(res_df)
    res_df.to_csv(out_csv, index=False)

    if out_valid_data is not None:
        with open(out_valid_path, 'wb') as f:
            pickle.dump(out_valid_data, f)


if __name__ == "__main__":
    main()
