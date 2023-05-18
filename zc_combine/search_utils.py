import random
from typing import List


def zc_warmup(df, func, n_warmup):
    assert n_warmup > 0
    zero_cost_pool = [random_net(df) for _ in range(n_warmup)]
    zero_cost_pool = func(zero_cost_pool)
    return zero_cost_pool


class NetData:
    def __init__(self, idx, df):
        self.idx = idx
        self.df = df

    def __str__(self):
        return self.val['net']

    @property
    def val(self):
        return self.df.loc[self.idx]

    def to_spec(self):
        return _get_spec_from_arch_str(self.val['net'])

    def get_proxy_score(self, proxy):
        return self.val[proxy]

    def get_val_acc(self):
        return self.val['val_accs']


def get_df_from_data(net_data: List[NetData]):
    df = net_data[0].df
    indices = []
    for nd in net_data:
        assert nd.df is df
        indices.append(nd.idx)

    return df.loc[indices]


def get_spec_map(df):
    return {v: i for i, v in df['net'].iteritems()}


def random_net(df):
    idx = random.choice(df.index)
    return NetData(idx, df)


def net_from_spec(spec, df, spec_map):
    arch_str = _spec_to_arch_str(spec)
    if arch_str not in spec_map:
        return None

    idx = spec_map[arch_str]
    return NetData(idx, df)


def _get_spec_from_arch_str(arch_str):
    spec = arch_str.strip('()').split(', ')
    return [int(i) for i in spec]


def _spec_to_arch_str(spec):
    return f"({', '.join([str(i) for i in spec])})"

