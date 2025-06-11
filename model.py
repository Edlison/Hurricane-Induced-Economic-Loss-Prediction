from collections import defaultdict

import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor


# ==== Metrics ====
def smape(y_true, y_pred):
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-8))


def rmsle(y_true, y_pred):
    y_true = np.maximum(0, y_true)
    y_pred = np.maximum(0, y_pred)
    return np.sqrt(np.mean(np.square(np.log1p(y_pred) - np.log1p(y_true))))


def evaluate_metrics(y_true_log, y_pred_log):
    """评估指标均在原始尺度下计算"""
    # TODO 直接log space 合理性
    # TODO y 通胀
    # y_true = np.expm1(y_true_log)
    y_true = y_true_log
    # y_pred = np.expm1(y_pred_log)
    y_pred = y_pred_log

    return {
        'R2': r2_score(y_true, y_pred),
        'MAE': mean_absolute_error(y_true, y_pred),
        'SMAPE': smape(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'RMSLE': rmsle(y_true, y_pred)
    }


# ==== Model Configuration ====
model_config = {
    'RF': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
    'XGB': XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42, verbosity=0),
    'NN': MLPRegressor(hidden_layer_sizes=(128,), alpha=0.001, max_iter=2000, early_stopping=True,
                       validation_fraction=0.1, random_state=42),
    'GBM': GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
}


# ==== Main Evaluation ====
def evaluate_one_model(X, y_log, model_name: str, seed: int = 42):
    assert model_name in model_config or model_name == 'Stacked', f"Invalid model: {model_name}"
    cv = KFold(n_splits=5, shuffle=True, random_state=seed)

    if model_name == 'Stacked':
        base_models = [(name, model_config[name]) for name in ['RF', 'XGB', 'NN', 'GBM']]
        model = StackingRegressor(estimators=base_models, final_estimator=Ridge(), cv=5)
    else:
        model = model_config[model_name]

    # 存储每一折的指标
    all_metrics = defaultdict(list)

    print(f"\nEvaluating {model_name} with 5-fold cross-validation:")
    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train_log, y_test_log = y_log[train_idx], y_log[test_idx]

        model.fit(X_train, y_train_log)
        y_pred_log = model.predict(X_test)

        metrics = evaluate_metrics(y_test_log, y_pred_log)
        for k, v in metrics.items():
            all_metrics[k].append(v)

        print(f"  Fold {fold_idx + 1} metrics: " + ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()]))

    # 汇总结果
    print(f"\n{model_name} performance (5-fold CV):")
    for k, v_list in all_metrics.items():
        mean_v = np.mean(v_list)
        std_v = np.std(v_list)
        print(f"{k}: {mean_v:.4f} ± {std_v:.4f}")

    return model
