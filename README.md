# zc_combine
Exploring combinations of zero-cost proxies

1. download all `zc_<searchspace>.json files`
- follow [README](https://github.com/automl/NASLib/tree/zerocost) in `zerocost` branch of `NASlib`
- copy files to `./data`

2. download `meta.json` for filtering unique nb201 nets
- from robustness benchmark [data](https://uni-siegen.sciebo.de/s/aFzpxCvTDWknpMA)
- download meta.zip, unzip to `./data/meta.json` 
- NOTE: aside from filtering using this filter, other isomorphic nets are also left out
  - e.g. (2, 1, 0, 1, 0, 1) and (2, 1, 0, 1, 0, 0) are the same, since the last op is 1 == zero op
  - see zc_combine/fixes/operations.py for edge and op explanation per search space
    - ex. for these two networks, the first position is the edge from node 1 to node 2, and the op is conv3x3

3. install necessary packages
- Develop branch of `naslib`
- `xgboost`


## Example runs

Setup env and go to the script dir:
```
pip install -e .
cd scripts
```

- fit a random forest on nb201 features, flops, params and jacov (sample size 1000)
```
python train_on_features.py --out_ OUT_DIR --benchmark nb201 --cfg ../zc_combine/configs/nb201_first.json \
    --meta ../data/meta.json --proxy jacov --model rf --train_size 1000
```

- fit xgboost on all nb201 proxies and no features (sample size 100)
```
python train_on_features.py --out_ OUT_DIR --benchmark nb201 --cfg ../zc_combine/configs/nb201_first.json \
    --meta ../data/meta.json --model xgb --use_all_proxies --no_features
```

## Run many experiments

1. create run config .json files
In `zc_combine/scripts/experiments`, there are folders for experiments. Every folder contains one or more .json files,
for example [train_args.json](https://github.com/gabikadlecova/zc_combine/blob/main/scripts/experiments/include_proxy_nb201/train_args.json):
```
{
  "data_seed": "42-47",
  "train_size": [20, 100, 1000],
  "cfg": ["../zc_combine/configs/nb201_first.json", "../zc_combine/configs/nb201_full.json"],
  "_raw_settings": ["--out_ train --meta ../data/meta.json"],
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
--data_seed 42 --train_size 20 --cfg "../zc_combine/configs/nb201_first.json" --out_ train --meta ../data/meta.json" \
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