# zc_combine
Exploring combinations of zero-cost proxies

1. download all `zc_<searchspace>.json files`
- follow [README](https://github.com/automl/NASLib/tree/zerocost) in `zerocost` branch of `NASlib`
- copy files to `./data`

2. download `meta.json` and robustness-data for filtering unique nb201 nets
- from robustness benchmark [data](https://uni-siegen.sciebo.de/s/aFzpxCvTDWknpMA)
- download meta.zip, unzip to `./data/robustness-dataset/meta.json`
- downlad all robustness data, untzip to `./data/robustness-dataset/{dataset}`
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
