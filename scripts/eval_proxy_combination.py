import os

import click
import matplotlib.pyplot as plt
import seaborn as sns

from scripts.utils import parse_proxy_settings, init_save_dir
from zc_combine.ensemble.eval import get_stats_ranks, eval_zerocost_score
from zc_combine.score import FilterProxyScore, SingleProxyScore, MeanScore
from zc_combine.utils.plot_utils import plot_networks_by_zc, plot_top_accuracy_zc, plot_filtered_by_zc, \
    plot_filtered_ranks
from zc_combine.utils.naslib_utils import load_search_space, parse_scores, load_search_spaces_multiple


sns.set()


def get_dfs(naslib_path, benchmark, dataset, plot_all):
    """Load one or several searchspaces. If `dataset is not None`, return only the dataset results for
       the search space(s)."""
    if plot_all:
        all_spaces = [os.path.splitext(f)[0] for f in os.listdir(os.path.join(naslib_path, 'naslib/data/')) if '.json' in f]
        zc_spaces = load_search_spaces_multiple(naslib_path, all_spaces, dataset_key=dataset)
        return parse_scores(zc_spaces)
    else:
        search_space = load_search_space(naslib_path, benchmark)
        dfs = parse_scores(search_space)
        return dfs if dataset is None else {dataset: dfs[dataset]}


def get_df_stats(dfs, filter_zc, rank_zc, quantile, mode, normalize):
    """Scores networks using either a mean of multiple proxies, or using a single proxy. If `filter_zc` is not None,
       score the networks using `filter_zc` and score only the top 1-quantile part."""

    rank_zc_str = '-'.join(rank_zc) if isinstance(rank_zc, list) else rank_zc
    name = f'{filter_zc}-{mode}-{quantile}-{rank_zc_str}' if filter_zc is not None else rank_zc_str

    def _score_nets(df):
        scorer = MeanScore(rank_zc, normalize=normalize) if isinstance(rank_zc, list) and len(rank_zc) >= 2 else SingleProxyScore(rank_zc)
        scorer = scorer if filter_zc is None else FilterProxyScore(filter_zc, scorer, quantile=quantile, mode=mode)
        scorer.fit(df)
        df[name] = scorer.predict(df)

    for df in dfs.values():
        _score_nets(df)

    return name, {task: eval_zerocost_score(df, name) for task, df in dfs.items() if name in df.columns}


def create_dirname(benchmark, plot_all, filter_zc, rank_zc, quantile, mode, dataset, minmax):
    prefix = benchmark if not plot_all else "all"
    if isinstance(filter_zc, str):
        filter_zc = [filter_zc]
    if isinstance(quantile, float):
        quantile = [quantile]
    filtstr = "" if filter_zc is None else '-'.join(filter_zc)
    filtstr = "" if filter_zc is None else f"filter-{filtstr}"
    quantstr = '-'.join([str(q) for q in quantile])
    datastr = "" if dataset is None else dataset
    modestr = "" if filter_zc is None or len(filter_zc) == 1 else mode
    rankstr = '-'.join(rank_zc) if isinstance(rank_zc, list) else rank_zc
    rankstr = f"rank-{rankstr}"
    minstr = 'minmax' if minmax else 'norm'
    minstr = '' if not isinstance(rank_zc, list) else minstr

    comps = [prefix, datastr, rankstr, filtstr, quantstr, modestr, minstr]
    comps = [c for c in comps if len(c)]
    return '_'.join(comps)


@click.command()
@click.argument('dir_path')
@click.argument('rank_zc')
@click.option('--benchmark', default=None)
@click.option('--dataset', default=None)
@click.option('--plot_all/--plot_one', default=False)
@click.option('--filter_zc', default=None)
@click.option('--naslib_path', default='../../zero_cost/NASLib')
@click.option('--quantile', default=0.8)
@click.option('--figsize', default="14,10")
@click.option('--mode', default='u')
@click.option('--key', default='corr')
@click.option('--minmax/--normalize', default=False)
def main(dir_path, rank_zc, benchmark, dataset, plot_all, filter_zc, naslib_path, quantile, figsize, mode, key, minmax):
    assert plot_all or benchmark is not None
    if plot_all:
        assert dataset is not None

    filter_zc, rank_zc, quantile = parse_proxy_settings(filter_zc, rank_zc, quantile)

    # create out dir
    save_path = create_dirname(benchmark, plot_all, filter_zc, rank_zc, quantile, mode, dataset, minmax)
    save_path = init_save_dir(dir_path, save_path)

    # process and filter
    dfs = get_dfs(naslib_path, benchmark, dataset, plot_all)
    rank_zc_colname, df_stats = get_df_stats(dfs, filter_zc, rank_zc, quantile, mode, not minmax)

    figsize = [int(f) for f in figsize.split(',')]

    if filter_zc is None:
        # Plot proxy scores of all networks, draw best networks (by accuracy) in orange, top 3 in green
        plot_networks_by_zc(df_stats, rank_zc_colname, benchmark, top_line=True, subplots_adjust=0.87,
                            zc_quantile=quantile, key=key, figsize=figsize)
        plt.savefig(os.path.join(save_path, 'full.png'))

        # Plot the best networks by accuracy
        plot_top_accuracy_zc(df_stats, rank_zc_colname, benchmark, subplots_adjust=0.87, zc_quantile=quantile,
                             key=key, figsize=figsize)
        plt.savefig(os.path.join(save_path, 'top_nets.png'))
    else:
        # Plot proxy scores of filtered networks
        plot_filtered_by_zc(df_stats, filter_zc, rank_zc_colname, benchmark if benchmark is not None else 'Search spaces',
                            quantile=quantile, key=key, figsize=figsize)
        plt.savefig(os.path.join(save_path, 'filter.png'))

    # If only a single search space, save stats and filtered/top network rank histograms.
    if not plot_all:
        stats, ranks = get_stats_ranks(df_stats)
        for data_name, rank_stats in ranks.items():
            for hist_key in ['rank', 'val_accs']:
                plot_filtered_ranks(rank_stats, benchmark, data_name, key=hist_key)
                plt.savefig(os.path.join(save_path, f'hist_{benchmark}_{data_name}_{hist_key}.png'))
        stats.to_csv(os.path.join(save_path, f'stats_{benchmark}.csv'))


if __name__ == "__main__":
    main()
