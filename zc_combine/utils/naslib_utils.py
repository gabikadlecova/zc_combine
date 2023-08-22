import glob
import json

import numpy as np
import pandas as pd


def list_search_spaces(spaces_path):
    return glob.glob(f'{spaces_path}/*.json')


def load_search_space(spaces_path, name):
    with open(f'{spaces_path}/{name}.json', 'r') as f:
        return json.load(f)


def load_search_spaces_multiple(naslib_path, names, dataset_key=None):
    all_data = {}
    for name in names:
        data = load_search_space(naslib_path, name)
        for k, v in data.items():
            if dataset_key is not None and k != dataset_key:
                continue
            all_data[f"{name}-{k}"] = v

    return all_data


def parse_scores(search_space, include_random=True):
    search_space_keys = [k for k in search_space.keys()]

    # get data of some random architecture
    arch_data = next(iter(search_space[search_space_keys[0]].values()))
    score_keys = [k for k in arch_data.keys() if k != 'id' and k != 'val_accuracy']

    # convert to DataFrame
    nets = {k: [] for k in search_space_keys}
    val_accs = {k: [] for k in search_space_keys}
    scores = {k: {s: [] for s in score_keys} for k in search_space_keys}

    # get only scores without running time
    for task in search_space.keys():
        for net, arch_scores in search_space[task].items():
            for s in score_keys:
                scores[task][s].append(arch_scores[s]['score'])

            val_accs[task].append(arch_scores['val_accuracy'])
            nets[task].append(net)

        # remove invalid scores
        for s in score_keys:
            if all([val == 0 for val in scores[task][s]]):
                scores[task].pop(s)

    # datasets
    dfs = {}
    for k in val_accs.keys():
        # random ordering
        if include_random:
            scores[k]['random'] = _generate_random_score(len(val_accs[k]))

        dfs[k] = pd.DataFrame({'net': nets[k], 'val_accs': val_accs[k], **scores[k]})
        dfs[k]['rank'] = dfs[k]['val_accs'].rank(ascending=False)

    return dfs


def _generate_random_score(length, seed=42):
    gen = np.random.default_rng(seed=seed)
    scores = np.arange(length)
    gen.shuffle(scores)
    return scores
