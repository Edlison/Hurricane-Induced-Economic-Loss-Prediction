from load_data import load_processed_data

if __name__ == '__main__':
    df_claims, df_hydro, df_storms = load_processed_data()
    print(df_claims.head())
    print(df_claims.shape)
    print(df_hydro.head())
    print(df_hydro.shape)
    print(df_storms.head())
    print(df_storms.shape)
