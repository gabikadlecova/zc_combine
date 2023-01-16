import glob
import json
import pandas as pd


def list_search_spaces(naslib_path):
    return glob.glob(f'{naslib_path}/naslib/data/*.json')


def load_search_space(naslib_path, name):
    with open(f'{naslib_path}/naslib/data/{name}.json', 'r') as f:
        return json.load(f)


def parse_scores(search_space):
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

    # datasets
    dfs = {}
    for k in val_accs.keys():
        dfs[k] = pd.DataFrame({'net': nets[k], 'val_accs': val_accs[k], **scores[k]})

    return dfs
