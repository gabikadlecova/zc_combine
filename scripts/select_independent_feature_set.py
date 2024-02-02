import argparse
import json
import math

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from zc_combine.utils.script_utils import load_or_create_features, get_net_data, load_bench_data, create_cache_filename

RANDOM_SEED=42

def create_features_dataset(searchspace_path, benchmark, dataset, cache_dir, cfg, version_key):
    
    data = load_bench_data(searchspace_path, benchmark, dataset, filter_nets=None, zero_unreachable=True)
    nets = get_net_data(data, benchmark)

    if cache_dir:
        cache_path = create_cache_filename(cache_dir, cfg, None, version_key, True)
    else:
        cache_path = None
        
    if cfg is not None:
        with open(cfg, 'r') as f:
            cfg = json.load(f)

    feature_dataset = load_or_create_features(nets, cfg, benchmark,
                                              features=None,
                                              cache_path=cache_path,
                                              version_key=version_key,
                                              compute_all=True)
    # for df in feature_dataset:
    #     print(df)
    return pd.concat(feature_dataset, axis=1)


def train_test_eval(data, feature, random_seed):

    
    X = data.drop(columns=[feature])
    y = data[feature]

    #    print("In features", len(X.columns))

    
    train_X, test_X, train_y, test_y = train_test_split(X, y, random_state=random_seed)

    model = LinearRegression()
    model.fit(train_X, train_y)

    score = model.score(test_X, test_y)
    
    return score

def evaluate(searchspace_path, benchmark, dataset, cache_dir, cfg, version_key,
             repeat=10, features=None, candidates=None):
    data = create_features_dataset(searchspace_path, benchmark, dataset, cache_dir, cfg, version_key)
    features = list(data.columns) if features is None else features
    data = data[features]

    
    candidates = features if candidates is None else candidates
    
    result = {f:[] for f in candidates} 
    for feature in candidates:
        for i in range(repeat):
            score = train_test_eval(data, feature, RANDOM_SEED+i)
            result[feature].append(score)
        print(feature, result[feature])

    return result 

def mean(r_list):
    return sum(r_list)/len(r_list)

def greedy_evaluate(data, repeat=10, features=None, candidates=None):


    features = list(data.columns) if features is None else features

    data = data[features]
    candidates = features if candidates is None else candidates

    dependent_features = []
    for feature in candidates:
        scores = [] 
        for i in range(repeat):
            score = train_test_eval(data, feature, RANDOM_SEED+i)
            scores.append(score)
        mean_score = mean(scores)
        if math.isclose(mean_score, 1.0):
            return feature, dependent_features
        else:
            dependent_features.append(feature)
            
    return None, dependent_features


def do_feature_selection(searchspace_path, benchmark, dataset, cache_dir, cfg, version_key):

    
    df_res = evaluate(searchspace_path, benchmark, dataset, cache_dir, cfg, version_key)
    features = list(df_res.keys())
    zero_features = [f for f in features if mean(df_res[f]) == 1.0]
    feature_to_delete = zero_features.pop()

    data = create_features_dataset(searchspace_path, benchmark, dataset, cache_dir, cfg, version_key)

    
    while zero_features: # and/or feature_to_delete is not None
        print("Removing feature: ", feature_to_delete)
        features = [f for f in features if f != feature_to_delete]
        print("Feature len:", len(features), flush=True)

        feature_to_delete, dependent_features = greedy_evaluate(data, features=features, candidates=zero_features)
        zero_features = [f for f in zero_features if f != feature_to_delete and f not in dependent_features]

    print(len(features))
    features = [ f.replace('[', '(').replace(']', ')') for f in features]
    for f in features:
        print(f)
    with open(f"{benchmark}_selected_features.json", "w") as f:
        json.dump(features, f) 
    
        
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Eliminate the linear dependent features."
    )
    parser.add_argument('--searchspace_path_', default='../data', help="Directory with json files of proxy scores "
                                                                      "(from NASLib).")
    parser.add_argument('--benchmark', default='nb201', help="Which NAS benchmark to use (e.g. nb201).")
    parser.add_argument('--data_seed', default=42, type=int, help="Seed for dataset splits.")
    parser.add_argument('--cache_dir_', default=None, help="Path to cache the feature datasets - filenames are composed "
                                                           "from cfg file name, features and version_key.")

    parser.add_argument('--cfg', default='../zc_combine/configs/nb201_full_short.json', type=str, help="Path to config file for proxy dataset creation. Example configs are `zc_combine/configs/*.json`.")

    parser.add_argument('--version_key', default=None, help="Version key of the cached dataset.")

    args = vars(parser.parse_args())
    
    searchspace_path = args['searchspace_path_']
    benchmark = args['benchmark']
    dataset = 'class_scene' if benchmark.startswith('tnb101') else 'cifar10'
    RANDOM_SEED = args['data_seed']
    cache_dir = args['cache_dir_']
    cfg = args['cfg']
    version_key = args['version_key']
    
    do_feature_selection(searchspace_path, benchmark, dataset, cache_dir, cfg, version_key)
    


