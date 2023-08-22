import click
import json

from utils import load_bench_data, get_net_data, load_uniques, get_bench_key
from zc_combine.features import feature_dicts
from zc_combine.features.dataset import get_feature_dataset


@click.command()
@click.option('--out', required=True)
@click.option('--benchmark', required=True, help="Possible values: nb101, nb201, tnb101, nb301.")
@click.option('--searchspace_path', default='../data')
@click.option('--dataset', default='cifar10')
@click.option('--config_path', required=True)
@click.option('--meta', default=None)
@click.option('--features', default=None, help="Optionally pass comma-separated list of feature function names to use "
                                               "from the config (default is use all of them).")
@click.option('--proxy', default=None)
@click.option('--use_all_proxies/--not_all_proxies', default=False)
@click.option('--use_features/--no_features', default=True)
@click.option('--use_flops_params/--no_flops_params', default=True)
def main(out, benchmark, searchspace_path, dataset, config_path, meta, features, proxy, use_all_proxies, use_features,
         use_flops_params):
    meta = load_uniques(meta) if meta is not None else meta

    data = load_bench_data(searchspace_path, benchmark, dataset, filter_nets=meta)
    nets = get_net_data(data, benchmark)

    with open(config_path, 'r') as f:
        cfg = json.load(f)

    # TODO load only if --use features
    features = features if features is None else features.split(',')
    feature_dataset = get_feature_dataset(nets, cfg, feature_dicts[get_bench_key(benchmark)], subset=features)

    # TODO add proxies like before

    # apply model (rf or xgb); eval tau, corr
    # udelat pca a feat importances (pres seedy do dataframu!)

    # add multiple prox for eval? maybe multiple scripts

    # TODO print lengths bef & after
    pass


if __name__ == "__main__":
    main()
