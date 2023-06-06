import os
import click
import matplotlib.pyplot as plt
import seaborn as sns

from zc_combine.ensemble.eval import eval_combined_proxies
from zc_combine.utils.naslib_utils import load_search_space, parse_scores

sns.set()


def plot_heatmap(name, df, key, quantile):
    inds, tau_scores = eval_combined_proxies(df, key=key, zc_quantile=quantile)
    names = [inds[i] for i in range(len(inds))]

    plt.figure(figsize=(12, 9))
    plt.title(f"{name} - {key}, quantile = {quantile}")
    sns.heatmap(tau_scores, annot=True, xticklabels=names, yticklabels=names)


@click.command()
@click.argument('dir_path')
@click.argument('benchmark')
@click.option('--naslib_path', default='../../zero_cost/NASLib')
@click.option('--quantile', default=0.8)
def main(dir_path, benchmark, naslib_path, quantile):
    """Evaluate all combinations of filter and rank proxies for `quantile`."""
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)

    save_path = f"heatmaps_{benchmark}_{quantile}"
    save_path = os.path.join(dir_path, save_path)
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    # process and filter
    search_space = load_search_space(naslib_path, benchmark)
    dfs = parse_scores(search_space)

    # plot heatmap
    for k, v in dfs.items():
        for key in ['tau', 'corr']:
            plot_heatmap(k, v, key, quantile)
            plt.savefig(os.path.join(save_path, f"heat_{k}_{key}.png"))


if __name__ == "__main__":
    main()
