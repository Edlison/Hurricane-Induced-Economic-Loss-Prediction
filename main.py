from features import prepare_features
from load_data import load_processed_data, load_zcta
from model import evaluate_one_model
from visualization import plot_feature_importance, plot_heatmap

# feature_names = ['building age', 'claim count', 'dam num', 'outlet num',
#                  'station num', 'streamgage num', 'wind speed', 'pressure',
#                  'hurricane scale', 'occupancy type',
#                  'occupancy type', 'occupancy type']
feature_names = ['num__buildingAge', 'num__Dam', 'num__Outlet', 'num__Station',
                 'num__numFloors', 'num__lowestFloorElevation', 'num__lowestAdjacentGrade',
                 'num__elevationDifference', 'num__elevatedBuildingIndicator', 'num__Streamgage', 'num__USA_WIND',
                 'num__USA_PRES', 'ordinal__USA_SSHS', 'cat__mainOccupancyType_1',
                 'cat__mainOccupancyType_11', 'cat__mainOccupancyType_4']


def run_prediction():
    df_claims, df_hydro, df_storms = load_processed_data()
    X, y, _ = prepare_features(df_claims, df_hydro, df_storms)
    model = evaluate_one_model(X, y, model_name='XGB')


def run_plot_importance():
    df_claims, df_hydro, df_storms = load_processed_data()
    X, y, _ = prepare_features(df_claims, df_hydro, df_storms)
    model = evaluate_one_model(X, y, model_name='XGB')
    plot_feature_importance(model, feature_names, model_name='XGBoost', top_k=12)


def run_plot_heatmap():
    df_claims, df_hydro, df_storms = load_processed_data()
    X, y, df_viz = prepare_features(df_claims, df_hydro, df_storms)
    print('viz columns: ', df_viz.columns)
    gdf_zcta = load_zcta()
    plot_heatmap(df_viz, gdf_zcta, feature='buildingCostSum')  # claimCount, USA_WIND, Dam, buildingCostSum


if __name__ == '__main__':
    run_plot_importance()
