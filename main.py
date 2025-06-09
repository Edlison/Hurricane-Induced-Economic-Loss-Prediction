from features import prepare_features
from load_data import load_processed_data, load_zcta
from model import evaluate_one_model
from visualization import plot_feature_importance, plot_heatmap

feature_names = ['num__buildingAge', 'num__claimCount', 'num__Dam', 'num__Outlet',
                 'num__Station', 'num__Streamgage', 'num__USA_WIND', 'num__USA_PRES',
                 'ordinal__USA_SSHS', 'cat__mainOccupancyType_1',
                 'cat__mainOccupancyType_11', 'cat__mainOccupancyType_4']


def run_prediction():
    df_claims, df_hydro, df_storms = load_processed_data()
    X, y, _ = prepare_features(df_claims, df_hydro, df_storms)
    model = evaluate_one_model(X, y, model_name='XGB')


def run_plot_importance():
    df_claims, df_hydro, df_storms = load_processed_data()
    X, y, _ = prepare_features(df_claims, df_hydro, df_storms)
    model = evaluate_one_model(X, y, model_name='XGB')
    plot_feature_importance(model, feature_names, model_name='XGBoost')


def run_plot_heatmap():
    df_claims, df_hydro, df_storms = load_processed_data()
    X, y, df_viz = prepare_features(df_claims, df_hydro, df_storms)
    gdf_zcta = load_zcta()
    plot_heatmap(df_viz, gdf_zcta, feature='Outlet')


if __name__ == '__main__':
    run_plot_heatmap()
