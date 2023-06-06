import os


def parse_proxy_settings(filter_zc, rank_zc, quantile):
    if filter_zc is not None and ',' in filter_zc:
        filter_zc = filter_zc.split(',')
    if ',' in rank_zc:
        rank_zc = rank_zc.split(',')
    if ',' in str(quantile):
        quantile = [float(q) for q in quantile.split(',')]
        assert isinstance(filter_zc, list) and len(quantile) == len(filter_zc)

    return filter_zc, rank_zc, quantile


def init_save_dir(experiments_dir, subdir):
    if not os.path.exists(experiments_dir):
        os.mkdir(experiments_dir)

    save_path = os.path.join(experiments_dir, subdir)
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    return save_path
