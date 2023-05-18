from typing import List, Union

import pandas as pd

from zc_combine.ensemble.filter import filter_by_zc
from zc_combine.search_utils import NetData, get_df_from_data, get_data_from_df


def score_nets(nets, proxy, filter_index=None, sort=True, pad_val=0.0):
    if filter_index is None and not isinstance(nets, pd.DataFrame):
        nets_scores = [(net.get_proxy_score(proxy), net) for net in nets]
        return sorted(nets_scores, key=lambda i: i[0], reverse=True) if sort else nets_scores

    nets = get_df_from_data(nets) if not isinstance(nets, pd.DataFrame) else nets
    nets_scores = nets[proxy].copy()

    if filter_index is not None:
        idx = nets.index.difference(filter_index)
        nets_scores.loc[idx] = pad_val

    return nets_scores.sort_values(ascending=False) if sort else nets_scores


class SingleProxyScore:
    def __init__(self, zc, sort=False):
        self.zc = zc
        self.sort = sort

    def fit(self, _: Union[List[NetData], pd.DataFrame]):
        return self

    def predict(self, nets: Union[List[NetData], pd.DataFrame]):
        return [n[1] for n in score_nets(nets, self.zc, sort=self.sort)]


class FilterProxyScore:
    def __init__(self, filter_zc: Union[str, List[str]], rank_zc: str, pad_val: float = 0.0,
                 quantile: Union[float, List[float]] = 0.8, mode='u', sort=False):

        self.filter_zc = [filter_zc] if isinstance(filter_zc, str) else filter_zc
        self.rank_zc = rank_zc
        self.quantile = quantile
        self.mode = mode

        self.fitted_quantiles = None
        if not isinstance(quantile, float):
            assert len(quantile) == len(filter_zc)

        self.pad_val = pad_val
        self.sort = sort

    def fit(self, nets: Union[List[NetData], pd.DataFrame]):
        if isinstance(nets, list):
            nets = get_df_from_data(nets)

        if isinstance(self.quantile, float):
            self.fitted_quantiles = [nets[fzc].quantile(self.quantile) for fzc in self.filter_zc]
            return

        self.fitted_quantiles = [nets[fzc].quantile(q) for q, fzc in zip(self.quantile, self.filter_zc)]

    def predict(self, nets: Union[List[NetData], pd.DataFrame]):
        net_df = get_df_from_data(nets) if isinstance(nets, list) else nets

        filter_index = filter_by_zc(net_df, self.filter_zc, self.fitted_quantiles, mode=self.mode)
        res = score_nets(net_df, self.rank_zc, filter_index=filter_index, sort=self.sort, pad_val=self.pad_val)
        return res if isinstance(nets, pd.DataFrame) else get_data_from_df(res.index, net_df.drop_duplicates())
