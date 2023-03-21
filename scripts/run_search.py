import os
import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from zc_combine.ea import run_evolution_search
from zc_combine.score import SingleProxyScore, FilterProxyScore
from zc_combine.utils.naslib_utils import load_search_space, parse_scores

sns.set()


@click.command()
@click.argument('dir_path')
@click.argument('rank_zc')
@click.argument('benchmark')
@click.argument('dataset')
@click.option('--filter_zc', default=None)
@click.option('--naslib_path', default='../../zero_cost/NASLib')
@click.option('--quantile', default=0.8)
@click.option('--mode', default='u')
@click.option('--n_times', default=10)
@click.option('--n_warmup', default=1000)
@click.option('--pool_size', default=64)
def main(dir_path, rank_zc, benchmark, dataset, filter_zc, naslib_path, quantile, mode, n_times, n_warmup, pool_size):
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)

    if filter_zc is not None and ',' in filter_zc:
        filter_zc = filter_zc.split(',')
    if ',' in str(quantile):
        quantile = [float(q) for q in quantile.split(',')]
        assert isinstance(filter_zc, list) and len(quantile) == len(filter_zc)

    if filter_zc is not None:
        filt_str = [filter_zc] if isinstance(filter_zc, str) else filter_zc
        filt_str = f"{'-'.join(filt_str)}_"
    else:
        filt_str = ''
    mod_str = f"_{mode}" if isinstance(filter_zc, list) else ''
    q_str = [str(q) for q in quantile] if isinstance(quantile, list) else [str(quantile)]
    q_str = f"{'-'.join(q_str)}"
    save_path = f"ea_{benchmark}_{dataset}_{filt_str}{rank_zc}_{q_str}{mod_str}_pool-{pool_size}"
    save_path = os.path.join(dir_path, save_path)
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    # process and filter
    search_space = load_search_space(naslib_path, benchmark)
    dfs = parse_scores(search_space)

    if filter_zc is None:
        ea_scoring = SingleProxyScore(rank_zc)
    else:
        ea_scoring = FilterProxyScore(dfs[dataset], filter_zc, rank_zc, quantile=quantile, mode=mode)

    all_res = []
    for n_exp in range(n_times):
        res = run_evolution_search(dfs[dataset], zc_warmup_func=ea_scoring, zero_cost_warmup=n_warmup,
                                   pool_size=pool_size, max_trained_models=500)
        all_res.extend([(n_exp, i, val) for i, val in enumerate(res)])
    all_res = pd.DataFrame(all_res, columns=['n_exp', 'T', 'val_acc'])
    name = f'rank: {rank_zc} filter: {filter_zc} q: {q_str}{mod_str}'
    all_res['name'] = name

    all_res.to_csv(os.path.join(save_path, 'log.csv'), index=False)


if __name__ == "__main__":
    main()
