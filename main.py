from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from load_data import load_processed_data


def prepare_features(df_claims, df_hydro, df_storm):
    # 合并三个 DataFrame
    df = df_claims.merge(df_hydro, on="ZCTA5CE20", how="left")
    df = df.merge(df_storm, on="ZCTA5CE20", how="left")

    # 目标变量
    y = df["buildingCostSum"].copy()

    # 特征选择：去除目标变量和ZCTA编码
    df_features = df.drop(columns=["ZCTA5CE20", "buildingCostSum"])

    # 类型识别
    numeric_features = df_features.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = df_features.select_dtypes(include=["object", "category"]).columns.tolist()

    # 构建预处理器
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
        ]
    )

    # 管道处理
    pipeline = Pipeline(steps=[("preprocessor", preprocessor)])
    X = pipeline.fit_transform(df_features)

    return X, y


if __name__ == '__main__':
    df_claims, df_hydro, df_storms = load_processed_data()
    X, y = prepare_features(df_claims, df_hydro, df_storms)
    print(X)
    print(X.shape, y.shape)
