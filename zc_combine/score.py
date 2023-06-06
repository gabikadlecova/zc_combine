from typing import List, Union

import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from zc_combine.ensemble.filter import filter_by_zc
from zc_combine.search_utils import NetData


def score_nets(nets, proxy, filter_index=None, sort=True, pad_val=0.0):
    """Score networks. If `filter_index` is provided, fill networks not in this index with `pad_val`."""
    nets_scores = nets[proxy].copy()

    if filter_index is not None:
        idx = nets.index.difference(filter_index)
        nets_scores.loc[idx] = pad_val

    return nets_scores.sort_values(ascending=False) if sort else nets_scores


class SingleProxyScore:
    """Score networks using a single zero-cost proxy."""
    def __init__(self, zc, sort=False):
        self.zc = zc
        self.sort = sort

    def fit(self, _: Union[List[NetData], pd.DataFrame]):
        return self

    def predict(self, nets: Union[List[NetData], pd.DataFrame], filter_index=None):
        return score_nets(nets, self.zc, sort=self.sort, filter_index=filter_index)


class FilterProxyScore:
    """Use a proxy (filter_zc) to filter out top score networks. Score these top networks using another scorer
       (rank_scorer)."""
    def __init__(self, filter_zc: Union[str, List[str]], rank_scorer, pad_val: float = 0.0,
                 quantile: Union[float, List[float]] = 0.8, mode='u', sort=False):

        self.filter_zc = [filter_zc] if isinstance(filter_zc, str) else filter_zc
        self.rank_scorer = rank_scorer
        self.quantile = quantile
        self.mode = mode

        self.fitted_quantiles = None
        if not isinstance(quantile, float):
            assert len(quantile) == len(filter_zc)

        self.pad_val = pad_val
        self.sort = sort

    def fit(self, nets: Union[List[NetData], pd.DataFrame]):
        self.rank_scorer.fit(nets)  # TODO some filter index?

        if isinstance(self.quantile, float):
            self.fitted_quantiles = [nets[fzc].quantile(self.quantile) for fzc in self.filter_zc]
            return

        self.fitted_quantiles = [nets[fzc].quantile(q) for q, fzc in zip(self.quantile, self.filter_zc)]
        return self

    def predict(self, nets: pd.DataFrame):
        filter_index = filter_by_zc(nets, self.filter_zc, self.fitted_quantiles, mode=self.mode)
        res = self.rank_scorer.predict(nets, filter_index=filter_index)
        return res


class MeanScore:
    """Score networks using several proxies. The resulting score is a sum of normalized scores."""
    def __init__(self, proxies: List[str], sort=False, normalize=True):
        self.proxies = proxies
        self.sort = sort

        self.scaler = StandardScaler() if normalize else MinMaxScaler()

    def fit(self, nets: pd.DataFrame):
        proxy_cols = nets[self.proxies]
        self.scaler.fit(proxy_cols)
        return self

    def predict(self, nets: pd.DataFrame, filter_index=None):
        proxy_cols = nets[self.proxies]
        proxy_cols = self.scaler.transform(proxy_cols)
        res = proxy_cols.sum(axis=1)

        colname = '-'.join(self.proxies)
        nets[colname] = res
        res = score_nets(nets, colname, filter_index=filter_index, sort=self.sort)
        nets.drop(columns=colname, inplace=True)

        return res
