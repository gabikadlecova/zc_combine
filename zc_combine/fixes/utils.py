import numpy as np
import pandas as pd


def read_new_score(data, new_data, proxy, copy_old_column=False, pad_val=0):
    data[f'new_{proxy}'] = data[proxy] if copy_old_column else pad_val

    for net_data in new_data:
        mask = data['net'] == net_data['arch']
        data_row = data[mask]
        assert len(data_row) == 1
        data.loc[mask, f'new_{proxy}'] = net_data[proxy]['score']


def aggregate_isomorphic_scores(data, proxy_cols, agg_func=np.median, val_accs_std=3, score_std=3):
    unique_data = []

    for net in data['new_net'].unique():
        nets = data[data['new_net'] == net]
        val_accs = nets['val_accs']

        agg = {s: agg_func(nets[s]) for s in proxy_cols}
        stds = {f"{s}_std": nets[s].std() for s in proxy_cols}
        assert len(nets) == 1 or all(s < score_std for s in stds.values())
        assert len(nets) == 1 or val_accs.std() < val_accs_std, (nets[['net', 'val_accs']], val_accs.std())

        unique_data.append({'net': net, 'val_accs': val_accs.median(), 'val_accs_std': val_accs.std(), **agg, **stds})

    return pd.DataFrame(unique_data)


def edge(edge_map, op_list, i, j):
    return op_list[edge_map[(i, j)]]


def zero_outgoing(net, edge_map, zero_op=1):
    for node in [2, 3]:
        inactive = all([(edge(edge_map, net, i, node) == zero_op) for i in range(1, node)])
        if not inactive:
            continue
        for j in range(node + 1, 5):
            net[edge_map[(node, j)]] = zero_op
    return net


def zero_ingoing(net, edge_map, zero_op=1):
    for node in [3, 2]:
        inactive = all([(edge(edge_map, net, node, j) == zero_op) for j in range(node + 1, 5)])
        if not inactive:
            continue
        for i in range(1, node):
            net[edge_map[(i, node)]] = zero_op
    return net


def nb201_zero_out_unreachable(data, edge_map, zero_op=1):
    for idx, net in zip(data.index, data['net']):
        net = [int(i) for i in net.strip('()').split(',')]
        net1 = zero_outgoing(net, edge_map, zero_op=zero_op)
        net2 = zero_ingoing(net1, edge_map, zero_op=zero_op)

        net = f"({', '.join([str(i) for i in net2])})"
        data.loc[idx, 'new_net'] = net
