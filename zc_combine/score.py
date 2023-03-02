from typing import List

from zc_combine.ensemble.filter import filter_by_zc
from zc_combine.search_utils import NetData


def score_nets(nets, proxy, sort=True):
    nets_scores = [(net.get_proxy_score(proxy), net) for net in nets]
    if sort:
        nets_scores = sorted(nets_scores, key=lambda i: i[0], reverse=True)
    return nets_scores


class SingleProxyScore:
    def __init__(self, zc, sort=True):
        self.zc = zc
        self.sort = sort

    def __call__(self, nets: List[NetData]):
        return score_nets(nets, self.zc, sort=self.sort)


class FilterProxyScore:
    def __init__(self, df, filter_zc, rank_zc, quantile=0.8, mode='u', sort=True):
        self.df = df

        self.filter_zc = filter_zc
        self.rank_zc = rank_zc
        self.quantile = quantile
        self.mode = mode

        self.sort = sort

    def __call__(self, nets: List[NetData]):
        indices = [net.idx for net in nets]
        nets_df = self.df.loc[indices]

        filter_idx = filter_by_zc(nets_df, self.filter_zc, quantile=self.quantile, mode=self.mode)
        filter_nets = [NetData(fi, self.df) for fi in filter_idx]

        return score_nets(filter_nets, self.rank_zc, sort=self.sort)
