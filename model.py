import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import make_scorer, mean_absolute_error, r2_score, mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor


# ==== Metrics ====
def smape(y_true, y_pred):
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-8))


def rmsle(y_true, y_pred):
    y_true = np.maximum(0, y_true)
    y_pred = np.maximum(0, y_pred)
    return np.sqrt(np.mean(np.square(np.log1p(y_pred) - np.log1p(y_true))))


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


scoring = {
    'MAE': make_scorer(mean_absolute_error),
    'R2': make_scorer(r2_score),
    'SMAPE': make_scorer(smape, greater_is_better=False),
    'RMSE': make_scorer(rmse),
    'RMSLE': make_scorer(rmsle, greater_is_better=False)
}

# ==== Model Configuration ====
model_config = {
    'RF': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
    'XGB': XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42, verbosity=0),
    'NN': MLPRegressor(hidden_layer_sizes=(100,), alpha=0.001, max_iter=1000, random_state=42),
    'GBM': GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
}


# ==== Main Evaluation ====
def evaluate_one_model(X, y, model_name: str, seed: int = 42):
    assert model_name in ['RF', 'XGB', 'NN', 'GBM', 'Stacked'], f"Invalid model: {model_name}"

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=seed)

    if model_name == 'Stacked':
        base_models = [(name, model_config[name]) for name in ['RF', 'XGB', 'NN', 'GBM']]
        model = StackingRegressor(estimators=base_models, final_estimator=Ridge(), cv=5)
    else:
        model = model_config[model_name]

    cv = KFold(n_splits=5, shuffle=True, random_state=seed)

    print(f"\nEvaluating model: {model_name}")
    for metric_name, scorer in scoring.items():
        scores = cross_val_score(model, X_train, y_train, scoring=scorer, cv=cv)
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        if "SMAPE" in metric_name or "RMSLE" in metric_name:  # loss-type
            mean_score = -mean_score
        print(f"{metric_name}: {mean_score:.4f} ± {std_score:.4f}")
