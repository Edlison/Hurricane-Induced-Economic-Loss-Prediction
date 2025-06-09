from features import prepare_features
from load_data import load_processed_data
from model import evaluate_one_model
from visualization import plot_feature_importance

feature_names = ['num__buildingAge', 'num__claimCount', 'num__Dam', 'num__Outlet',
                 'num__Station', 'num__Streamgage', 'num__USA_WIND', 'num__USA_PRES',
                 'ordinal__USA_SSHS', 'cat__mainOccupancyType_1',
                 'cat__mainOccupancyType_11', 'cat__mainOccupancyType_4']

if __name__ == '__main__':
    df_claims, df_hydro, df_storms = load_processed_data()
    X, y = prepare_features(df_claims, df_hydro, df_storms)
    model = evaluate_one_model(X, y, model_name='XGB')
    plot_feature_importance(model, feature_names, model_name='XGBoost')
