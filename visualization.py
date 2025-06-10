import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import geopandas as gpd


def plot_feature_importance(model, feature_names, model_name: str, top_k: int = 20):
    """
    Plot feature importances for supported models.

    Parameters:
        model: fitted model instance
        feature_names: list or array of feature names
        model_name: string name of the model (for title)
        top_k: number of top features to display
    """
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 14  # Set base font size

    if hasattr(model, "feature_importances_"):  # Tree-based models
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):  # Linear models
        importances = np.abs(model.coef_)
        if importances.ndim > 1:  # For multi-output or multi-task models
            importances = importances[0]
    else:
        raise ValueError(f"Model type {type(model)} does not support feature importance.")

    # Sort and select top_k
    feature_importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False).head(top_k)

    # Plot
    plt.figure(figsize=(8, 6))
    plt.barh(feature_importance_df["Feature"][::-1], feature_importance_df["Importance"][::-1])
    plt.xlabel("Importance", fontsize=22)
    plt.title(f"Top {top_k} Feature Importances ({model_name})", fontsize=22)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.tight_layout()
    plt.savefig(f"./imgs/plot_importance_{model_name}.pdf", format="pdf", dpi=300)
    plt.show()


def plot_heatmap(df_viz: pd.DataFrame, gdf_zcta: gpd.GeoDataFrame, feature: str,
                 cmap: str = 'OrRd') -> None:
    """
    根据指定 feature 生成 ZCTA 区域热力图。

    参数:
        df_viz (pd.DataFrame): 包含 ZCTA5CE20 和特征列的数据。
        gdf_zcta (gpd.GeoDataFrame): 包含 ZCTA5CE20 和 geometry 的地理数据。
        feature (str): 想要绘图的特征列名。
        cmap (str): 颜色映射方案，默认 'OrRd'。可选：'YlGnBu', 'viridis', 'plasma', 'Reds'
        title (str): 图标题，默认根据 feature 自动生成。
        figsize (tuple): 图尺寸，默认 (10, 10)。
    """
    # feature map
    feature_map = {'USA_WIND': 'Wind speed', 'claimCount': 'Claim count', 'Dam': 'Dam num', 'buildingCostSum': 'Replacement cost'}
    # 类型转换，确保合并时一致
    df_viz['ZCTA5CE20'] = df_viz['ZCTA5CE20'].astype(str)
    gdf_zcta['ZCTA5CE20'] = gdf_zcta['ZCTA5CE20'].astype(str)

    # 合并数据
    gdf_merged = gdf_zcta.merge(df_viz[['ZCTA5CE20', feature]], on='ZCTA5CE20', how='left')

    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 22

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    gdf_merged.plot(
        column=feature,
        cmap=cmap,
        linewidth=0.4,
        edgecolor='grey',
        legend=True,
        ax=ax,
        legend_kwds={
            'shrink': 0.8,
            'aspect': 20,
            'label': feature_map.get(feature, ''),
        }
    )

    ax.set_title(f"{feature_map.get(feature, '')} Heatmap by ZCTA in Florida", fontsize=22)
    ax.axis('off')

    # 调整 colorbar tick 和 label 字体
    cbar = fig.axes[-1]
    cbar.tick_params(labelsize=20)
    cbar.set_ylabel(feature_map.get(feature, ''), fontsize=22)
    plt.tight_layout()
    plt.savefig(f"./imgs/plot_heatmap_{feature}.pdf", format="pdf", dpi=300)
    plt.show()
