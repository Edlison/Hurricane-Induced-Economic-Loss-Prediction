from features import prepare_features
from load_data import load_processed_data, load_zcta
from model import evaluate_one_model
from visualization import plot_feature_importance, plot_feature_importance_xtick, plot_heatmap, plot_heatmap_grid

# feature_names = ['building age', 'claim count', 'dam num', 'outlet num',
#                  'station num', 'streamgage num', 'wind speed', 'pressure',
#                  'hurricane scale', 'occupancy type',
#                  'occupancy type', 'occupancy type']
feature_names = ['building age', 'floors', 'elevation diff',
                 'lowest floor', 'lowest adjacent',
                 'elevated buildings', 'dam', 'outlet', 'station',
                 'streamgage', 'wind speed', 'pressure', 'hurricane scale',
                 'occupancy type', 'occupancy type',
                 'occupancy type']


def run_prediction():
    df_claims, df_hydro, df_storms = load_processed_data()
    X, y, _ = prepare_features(df_claims, df_hydro, df_storms)
    model = evaluate_one_model(X, y, model_name='Stacked')


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


if __name__ == '__main__':
    run_plot_heatmap()
