import click
import json

from utils import load_bench_data, get_net_data, load_uniques, get_dataset, get_data_splits, eval_model
from zc_combine.predictors import predictor_cls


@click.command()
@click.option('--out', required=True)
@click.option('--benchmark', required=True, help="Possible values: nb101, nb201, tnb101, nb301.")
@click.option('--searchspace_path', default='../data')
@click.option('--dataset', default='cifar10')
@click.option('--cfg', default=None)
@click.option('--meta', default=None)
@click.option('--features', default=None, help="Optionally pass comma-separated list of feature function"
                                               "names to use from the config (default is use all of them).")
@click.option('--proxy', default=None)
@click.option('--use_all_proxies/--not_all_proxies', default=False)
@click.option('--use_features/--no_features', default=True)
@click.option('--use_flops_params/--no_flops_params', default=True)
@click.option('--n_evals', default=10, help="Number of models fitted and evaluated on the data (with random"
                                            "state seed + i).")
@click.option('--seed', default=42, help="Starting seed.")
@click.option('--data_seed', default=42, help="Data split seed.")
@click.option('--train_size', default=100, help="Number of train architectures sampled.")
@click.option('--model', default='rf', help="Model to use (rf, xgb, xgb_tuned).")
def main(out, benchmark, searchspace_path, dataset, cfg, meta, features, proxy, use_all_proxies, use_features,
         use_flops_params, n_evals, seed, data_seed, train_size, model):

    # load meta.json to filter unique nets from nb201 and tnb101
    meta = load_uniques(meta) if meta is not None else meta

    data = load_bench_data(searchspace_path, benchmark, dataset, filter_nets=meta)
    nets = get_net_data(data, benchmark)

    if cfg is not None:
        with open(cfg, 'r') as f:
            cfg = json.load(f)

    features = features if features is None else features.split(',')
    proxy = proxy.split(',') if proxy is not None else []

    dataset, y = get_dataset(data, nets, benchmark, cfg=cfg, features=features, proxy_cols=proxy,
                             use_features=use_features,
                             use_all_proxies=use_all_proxies,
                             use_flops_params=use_flops_params)

    data_splits = get_data_splits(dataset, y, random_state=data_seed, train_size=train_size)

    model_cls = predictor_cls[model]
    fitted_models, res = eval_model(model_cls, data_splits, n_times=n_evals, random_state=seed)

    # TODO wandb?
    # udelat pca a feat importances (pres seedy do dataframu!)

    # TODO print lengths bef & after (uniques)

    # TODO doplnit readme co je treba stahnout, nainstalovat naslib,...
    pass


if __name__ == "__main__":
    main()
