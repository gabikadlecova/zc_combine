import os
import click
import pandas as pd
import seaborn as sns

from utils import parse_proxy_settings, init_save_dir
from zc_combine.ea import run_evolution_search
from zc_combine.score import SingleProxyScore, FilterProxyScore, MeanScore
from zc_combine.utils.naslib_utils import load_search_space, parse_scores

sns.set()


def create_names(filter_zc, rank_zc, mode, quantile, benchmark, dataset, pool_size):
    if filter_zc is not None:
        filt_str = [filter_zc] if isinstance(filter_zc, str) else filter_zc
        filt_str = f"{'-'.join(filt_str)}_"
    else:
        filt_str = ''
    rank_str = '-'.join(rank_zc) if isinstance(rank_zc, list) else rank_zc
    mod_str = f"_{mode}" if isinstance(filter_zc, list) else ''
    q_str = [str(q) for q in quantile] if isinstance(quantile, list) else [str(quantile)]
    q_str = f"{'-'.join(q_str)}"

    dir_name = f"ea_{benchmark}_{dataset}_f-{filt_str}r-{rank_str}_{q_str}{mod_str}_pool-{pool_size}"
    stats_name = f'rank: {rank_str} filter: {filt_str} q: {q_str}{mod_str}'

    return dir_name, stats_name


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
    # parse path and settings
    filter_zc, rank_zc, quantile = parse_proxy_settings(filter_zc, rank_zc, quantile)

    save_path, name = create_names(filter_zc, rank_zc, mode, quantile, benchmark, dataset, pool_size)
    save_path = init_save_dir(dir_path, save_path)

    # process and filter
    search_space = load_search_space(naslib_path, benchmark)
    dfs = parse_scores(search_space)

    # init scoring and run algo
    ea_scoring = MeanScore(rank_zc) if isinstance(rank_zc, list) and len(rank_zc) > 1 else SingleProxyScore(rank_zc)
    if filter_zc is not None:
        ea_scoring = FilterProxyScore(filter_zc, ea_scoring, quantile=quantile, mode=mode)

    all_res = []
    for n_exp in range(n_times):
        res = run_evolution_search(dfs[dataset], zc_warmup_func=ea_scoring, zero_cost_warmup=n_warmup,
                                   pool_size=pool_size, max_trained_models=500)
        all_res.extend([(n_exp, i, val) for i, val in enumerate(res)])
    all_res = pd.DataFrame(all_res, columns=['n_exp', 'T', 'val_acc'])
    all_res['name'] = name

    all_res.to_csv(os.path.join(save_path, 'log.csv'), index=False)


if __name__ == "__main__":
    main()
