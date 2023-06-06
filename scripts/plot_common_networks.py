import os
import click
import matplotlib.pyplot as plt
import seaborn as sns

from scripts.utils import init_save_dir
from zc_combine.ensemble.filter import common_n_largest
from zc_combine.utils.naslib_utils import load_search_space, parse_scores
from zc_combine.utils.plot_utils import plot_common_networks

sns.set()


@click.command()
@click.argument('dir_path')
@click.argument('benchmark')
@click.option('--naslib_path', default='../../zero_cost/NASLib')
@click.option('--n_largest', default=50)
def main(dir_path, benchmark, naslib_path, n_largest):
    """Plot for searchspace pairs their common best networks (best ... among `n_largest` networks)."""

    save_path = f"common-nets_{benchmark}_{n_largest}"
    save_path = init_save_dir(dir_path, save_path)

    # process and filter
    search_space = load_search_space(naslib_path, benchmark)
    dfs = parse_scores(search_space)

    # plot heatmap
    inds, common_nets = common_n_largest(dfs, n_largest=n_largest)

    plot_common_networks(common_nets, inds, n_largest=n_largest)
    plt.savefig(os.path.join(save_path, 'common.png'))


if __name__ == "__main__":
    main()
