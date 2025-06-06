import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def prepare_features(df_claims, df_hydro, df_storm):
    # 合并三个 DataFrame
    df = df_claims.merge(df_hydro, on="ZCTA5CE20", how="left")
    df = df.merge(df_storm, on="ZCTA5CE20", how="left")

    # 目标变量
    y = df["buildingCostSum"].copy()

    # Feature Engineering
    df["buildingAge"] = 2025 - df["avgConstructionYear"]
    df["mainOccupancyType"] = df["mainOccupancyType"].astype("Int64").astype(str)

    # 可选 log 变换（防止偏态）
    for col in ["Dam", "Outlet", "Station", "Streamgage", "claimCount"]:
        df[col] = np.log1p(df[col])  # log(1 + x)

    # 明确选择特征列
    feature_cols = [
        "buildingAge", "mainOccupancyType", "claimCount",
        "Dam", "Outlet", "Station", "Streamgage",
        "USA_WIND", "USA_SSHS", "USA_PRES"
    ]
    df_features = df[feature_cols]

    # for col in df_features.columns:
    #     print(f"{col}: {df_features[col].dtype}")
    # print(df_features.columns)
    # print(df_features.head())
    # print(df_features.describe())
    print(y.describe())

    # 分类特征与数值特征明确划分
    numeric_features = [
        "buildingAge", "claimCount", "Dam", "Outlet", "Station",
        "Streamgage", "USA_WIND", "USA_PRES"
    ]
    no_scale_features = ["USA_SSHS"]
    categorical_features = ["mainOccupancyType"]

    # 构建预处理器
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("ordinal", "passthrough", no_scale_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
        ]
    )

    # 管道处理
    pipeline = Pipeline(steps=[("preprocessor", preprocessor)])
    X = pipeline.fit_transform(df_features)

    return X, np.log1p(y)  # log1p y
