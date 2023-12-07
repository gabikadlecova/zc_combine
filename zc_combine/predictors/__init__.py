from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor

xgb_args = {
    "tree_method": "hist",
    "subsample": 0.9,
    "n_estimators": 10000,
    "learning_rate": 0.01
}

predictor_cls = {
    'rf': lambda seed, **kwargs: RandomForestRegressor(random_state=seed, **kwargs),
    'xgb': lambda seed, **kwargs: XGBRegressor(random_state=seed, **kwargs),
    'xgb_tuned': lambda seed: XGBRegressor(random_state=seed, **xgb_args),
    'mlp': lambda seed: MLPRegressor([90, 180, 180], learning_rate_init=0.01, max_iter=1000)
}
