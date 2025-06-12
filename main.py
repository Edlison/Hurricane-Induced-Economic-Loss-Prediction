import argparse

from features import prepare_features
from load_data import load_processed_data, load_zcta
from model import evaluate_one_model
from visualization import plot_feature_importance, plot_feature_importance_xtick, plot_heatmap, plot_heatmap_grid, \
    plot_heatmap_2

feature_names = ['building age', 'floors', 'elevation diff',
                 'lowest floor', 'lowest adjacent',
                 'elevated buildings', 'dam', 'outlet', 'station',
                 'streamgage', 'wind speed', 'pressure', 'hurricane scale',
                 'occupancy type', 'occupancy type',
                 'occupancy type']


def run_prediction(model_name):
    df_claims, df_hydro, df_storms = load_processed_data()
    X, y, _ = prepare_features(df_claims, df_hydro, df_storms)
    model = evaluate_one_model(X, y, model_name)


def run_plot_importance():
    df_claims, df_hydro, df_storms = load_processed_data()
    X, y, _ = prepare_features(df_claims, df_hydro, df_storms)
    model = evaluate_one_model(X, y, model_name='XGB')
    plot_feature_importance(model, feature_names, model_name='XGBoost', top_k=10)


def run_plot_importance_xtick():
    df_claims, df_hydro, df_storms = load_processed_data()
    X, y, _ = prepare_features(df_claims, df_hydro, df_storms)
    model = evaluate_one_model(X, y, model_name='XGB')
    plot_feature_importance_xtick(model, feature_names, model_name='XGBoost', top_k=10)


def run_plot_heatmap():
    df_claims, df_hydro, df_storms = load_processed_data()
    X, y, df_viz = prepare_features(df_claims, df_hydro, df_storms)
    print('viz columns: ', df_viz.columns)
    gdf_zcta = load_zcta()
    plot_heatmap(df_viz, gdf_zcta,
                 feature='Dam')  # elevatedBuildingIndicator, USA_WIND, Dam, totalCostInflated


def run_plot_heatmap_grid():
    df_claims, df_hydro, df_storms = load_processed_data()
    X, y, df_viz = prepare_features(df_claims, df_hydro, df_storms)
    print('viz columns: ', df_viz.columns)
    gdf_zcta = load_zcta()
    plot_heatmap_grid(df_viz, gdf_zcta, features=[
        'elevatedBuildingIndicator', 'USA_WIND', 'Dam', 'totalCostInflated'
    ])


def run_plot_heatmap_2():
    df_claims, df_hydro, df_storms = load_processed_data()
    X, y, df_viz = prepare_features(df_claims, df_hydro, df_storms)
    print('viz columns: ', df_viz.columns)
    gdf_zcta = load_zcta()
    plot_heatmap_2(df_viz, gdf_zcta)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run ML model for hurricane loss prediction')
    parser.add_argument('--model_name', type=str, required=True,
                        help='Name of the model to run (e.g., RF, XGB, NN, Stacked)')
    args = parser.parse_args()

    run_prediction(model_name=args.model_name)
