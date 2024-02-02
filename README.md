# zc_combine
Exploring combinations of zero-cost proxies

1. download all `zc_<searchspace>.json files`
- follow [README](https://github.com/automl/NASLib/tree/zerocost) in `zerocost` branch of `NASlib`
- copy files to `./data`

2. install necessary packages
- Develop branch of `naslib`
- `xgboost`

### Data used in prediction NB201 and TNB101_micro
Networks with branches with unreachable operations (due to zero operations) are left out
- e.g. (2, 1, 0, 1, 0, 2) and (2, 1, 0, 1, 0, 1) are the same - although there is a convolution between nodes 3 and 4
  (the last number is 2), it gets no input, since edges 1-3 and 2-3 are zero (number 1 in this encoding)
- see zc_combine/fixes/operations.py for edge and op explanation per search space
    - ex. for these two networks, the first position is the edge from node 1 to node 2, and the op is conv3x3


## Example runs

Setup env and go to the script dir:
```
pip install -e .
cd scripts
```

- fit a random forest on nb201 features, flops, params and jacov (sample size 1000)
```
python train_on_features.py --out_ OUT_DIR --benchmark nb201 --cfg ../zc_combine/configs/nb201_first.json \
    --proxy jacov --model rf --train_size 1000
```

- fit xgboost on all nb201 proxies and no features (sample size 100)
```
python train_on_features.py --out_ OUT_DIR --benchmark nb201 --cfg ../zc_combine/configs/nb201_first.json \
    --model xgb --use_all_proxies --no_features
```

## Run many experiments

1. create run config .json files
In `zc_combine/scripts/experiments`, there are folders for experiments. Every folder contains one or more .json files,
for example train_args.json ... ./scripts/experiments/include_proxy_nb201/train_args.json:
```
{
  "data_seed": "42-47",
  "train_size": [20, 100, 1000],
  "cfg": ["../zc_combine/configs/nb201_first.json", "../zc_combine/configs/nb201_full.json"],
  "_raw_settings": ["--out_ train"],
  "_raw_columns": [
    "--use_all_proxies", "",
    "--proxy epe_nas", "--proxy fisher",  "--proxy grad_norm", "--proxy grasp", "--proxy jacov", "--proxy l2_norm",
    "--proxy nwot", "--proxy plain", "--proxy snip", "--proxy synflow", "--proxy zen"],
  "model": ["rf"],
  "wandb_project_": "zc_nb201"
}
```

- `"model": ["rf"]` will be converted to `--model rf`
- values "42-47" are equivalent to [42, 43, 44, 45, 46]
- lists of values are all possible settings for the command line option
- entries with keys prefixed by "_raw" will be passed without the key name
    - e.g. `"_raw_proxy": "--proxy jacov,fisher"` would be passed as `--proxy jacov,fisher`

2. create all possible runs
The script `zc_combine/scripts/create_args.py` takes as input one experiment directory and produces a cartesian product
of provided arguments (all possible combinations of arg settings). These are saved to a .csv file, where
every row is one run setting, and every column is a list of command line args created from the config.

For example, the first run setting created from the above .json will be:
```
--data_seed 42 --train_size 20 --cfg "../zc_combine/configs/nb201_first.json" --out_ train  \
--use_all_proxies --model rf --wandb_project_ zc_nb201
```

3. Run every K-th row 
The script `zc_combine/scripts/run_with_multiple_args.sh` serves for running all rows of `experiment_settings.csv` in
multiple calls (e.g. in tmux or as batch jobs).
The following example call:
- activates environment `env` (modify script for conda envs)
- runs train and evaluation on `(i + K * j)`-th rows, `(j = 0, 1, 2...)`
- logs config and results to wandb
- caches feature dataset in directory ./cache_data/ (to avoid computing the same features every time)

```
bash run_with_multiple_args.sh experiments/<experiment dir>/experiment_settings.csv i K bash run_train.sh <env_name> \
     <wandb_key> ./cache_data/ <cache data version>
```

4. Extensions to run scripts
- In the previous call, you can replace `run_train.sh` by any similar script.
- `run_with_multiple_args.sh` reads column values of `experiment_settings.csv` and passes them as environment variables
  to `run_train.sh` (or any other script)
    - e.g. `train_args.json` -> column name `train_args` -> set env. var. `train_args="--model rf --data_seed 42"`
- `run_pca_feature_selection.sh` works similarly to `run_train.sh`, but needs three json files instead of one. Refer to 
  `zc_combine/scripts/experiments/pca_*/` for example config files
    - Dataset config args should be specified in the pca json file to ensure running with the same data.
      **Avoid specifying any dataset args in the train json file**


## HW experiments

1. HW files

 - download `HW-NAS-Bench-v1_0.pickle` from https://github.com/GATECH-EIC/HW-NAS-Bench
 - save it do `data/`
 - `python prepare_hw_bench.py`

2. Config files to run HW experiments:

experiments0/paper_hw_cifar10/
experiments/paper_hw_cifar100/
experiments/paper_hw_imagenet/

3. Config files to run RF acc. prediction with logging importances:
(has to be run localy)

```
 experiments/paper_nb101/train_args.json
 experiments/paper_nb201/train_args.json
 experiments/paper_nb301/train_args.json
 experiments/paper_tnb101_macro/train_args.json
 experiments/paper_tnb101_micro/train_args.json
```
Example:
`python create_args_df.py --dir_path experiments/paper_nb201/`
`bash run_with_multiple_args.sh experiments/paper_nb201/experiment_settings.csv 0 1 bash run_train_local.sh $ENV_NAME paper_exp ./cache/ paper_version`

4.1. Analyzing importances:

`cd paper_exp`
`python ../analyze_importances.py 1024 nb201 cifar10`

This produces CSV files named `{benchmark}_{dataset}_{train_size}.csv`.

4.2. Creating tables:

Go to directory `results` and copy CSV files produced in previous step to `data/`.

`python important_features_tables.py`
`./table.sh`


5. Independent set of features selection:

```usage: select_independent_feature_set.py [-h]
                                         [--searchspace_path_ SEARCHSPACE_PATH_]
                                         [--benchmark BENCHMARK]
                                         [--data_seed DATA_SEED]
                                         [--cache_dir_ CACHE_DIR_] [--cfg CFG]
                                         [--version_key VERSION_KEY]```

To create linearly independent sets of features run:
`python select_independent_feature_set.py --benchmark nb101 --cfg ../zc_combine/configs/nb101_first_short.json`
`python select_independent_feature_set.py --benchmark nb201 --cfg ../zc_combine/configs/nb201_full_short.json`
`python select_independent_feature_set.py --benchmark nb301 --cfg ../zc_combine/configs/nb301_full_short.json`

It outputs `{benchmark}_selected_features.json`.

6. Run predictions with selected features only:

Use config files in directories:
```
experiments/nb101_selected/
experiments/nb201_selected/
experiments/nb301_selected/
```

