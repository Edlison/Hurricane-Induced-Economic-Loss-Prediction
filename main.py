from features import prepare_features
from load_data import load_processed_data
from model import evaluate_one_model

if __name__ == '__main__':
    df_claims, df_hydro, df_storms = load_processed_data()
    X, y = prepare_features(df_claims, df_hydro, df_storms)
    results = evaluate_one_model(X, y, model_name='XGB')
