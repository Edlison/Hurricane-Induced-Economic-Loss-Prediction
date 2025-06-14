import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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


def plot_feature_importance_xtick(model, feature_names, model_name: str, top_k: int = 20):
    """
    Plot feature importances with features on x-axis and 30-degree rotated labels.

    Parameters:
        model: fitted model instance
        feature_names: list or array of feature names
        model_name: string name of the model (for title)
        top_k: number of top features to display
    """
    import matplotlib.cm as cm

    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 14
    oranges_color = cm.get_cmap('OrRd')(0.7)

    if hasattr(model, "feature_importances_"):  # Tree-based models
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):  # Linear models
        importances = np.abs(model.coef_)
        if importances.ndim > 1:
            importances = importances[0]
    else:
        raise ValueError(f"Model type {type(model)} does not support feature importance.")

    # Sort and select top_k
    feature_importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False).head(top_k)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.bar(feature_importance_df["Feature"], feature_importance_df["Importance"], color=oranges_color)
    plt.title(f"Top {top_k} Feature Importances ({model_name})", fontsize=24)
    plt.ylabel("Importance", fontsize=24)
    plt.xticks(rotation=30, ha='right', fontsize=24)
    plt.yticks(fontsize=24)
    plt.tight_layout()
    plt.savefig(f"./imgs/plot_importance_{model_name}_xtick.pdf", format="pdf", dpi=600)
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
    feature_map = {'USA_WIND': 'Wind speed', 'claimCount': 'Claim count', 'Dam': 'Dams',
                   'buildingCostSum': 'Replacement cost', 'totalCostInflated': 'Inflated cost (log(1 + x) USD)'}
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

    # ax.set_title(f"{feature_map.get(feature, '')} Heatmap by ZCTA in Florida", fontsize=22)
    ax.axis('off')

    # 调整 colorbar tick 和 label 字体
    cbar = fig.axes[-1]
    cbar.tick_params(labelsize=20)
    cbar.set_ylabel(feature_map.get(feature, ''), fontsize=22)
    plt.tight_layout()
    plt.savefig(f"./imgs/plot_heatmap_{feature}.pdf", format="pdf", bbox_inches='tight', dpi=300)
    plt.show()


def plot_heatmap_grid(df_viz: pd.DataFrame, gdf_zcta: gpd.GeoDataFrame, features: list[str]) -> None:
    """
    在同一图中绘制多个特征的 ZCTA 热力图（2行2列）。

    参数:
        df_viz (pd.DataFrame): 包含 ZCTA5CE20 和特征列的数据。
        gdf_zcta (gpd.GeoDataFrame): 包含 ZCTA5CE20 和 geometry 的地理数据。
        features (list[str]): 想要绘图的特征列名列表。
        cmap (str): 颜色映射方案。
        figsize (tuple): 图尺寸。
    """
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    feature_map = {
        'USA_WIND': 'Wind speed',
        'Dam': 'Dam count',
        'elevatedBuildingIndicator': 'Elevated Building Indicator',
        'totalCostInflated': 'Total Inflated Cost'
    }

    plot_map = {
        'elevatedBuildingIndicator': '(a)',
        'USA_WIND': '(b)',
        'Dam': '(c)',
        'totalCostInflated': '(d)'
    }

    df_viz['ZCTA5CE20'] = df_viz['ZCTA5CE20'].astype(str)
    gdf_zcta['ZCTA5CE20'] = gdf_zcta['ZCTA5CE20'].astype(str)
    gdf_merged = gdf_zcta.merge(df_viz[['ZCTA5CE20'] + features], on='ZCTA5CE20', how='left')

    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 18

    fig, axes = plt.subplots(2, 2, figsize=(16, 12), gridspec_kw={'hspace': 0.01, 'wspace': 0.01})
    axes = axes.flatten()

    for i, feature in enumerate(features):
        ax = axes[i]
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)

        gdf_merged.plot(
            column=feature,
            cmap='OrRd',
            linewidth=0.3,
            edgecolor='grey',
            ax=ax,
            legend=True,
            cax=cax,
            legend_kwds={
                'label': feature_map.get(feature, feature)
            }
        )

        # ax.set_title(f"{feature_map.get(feature, feature)} by ZCTA", fontsize=18)
        ax.text(0.5, -0.01, plot_map.get(feature, feature), fontsize=28,
                ha='center', va='center', transform=ax.transAxes)
        ax.axis('off')
        cax.tick_params(labelsize=20)
        cax.set_ylabel(feature_map.get(feature, feature), fontsize=25)

    fig.subplots_adjust(left=0.0, right=0.96, top=0.98, bottom=0.02)
    plt.savefig("./imgs/plot_heatmap_grid.pdf", format="pdf", dpi=300)
    plt.show()


def plot_heatmap_2(df_viz: pd.DataFrame, gdf_zcta: gpd.GeoDataFrame) -> None:
    """
    并列绘制两个特征的 ZCTA 区域热力图：totalCostInflated 与 Dam，
    colorbar 纵向浮动在图的左下角，长度为原来的 0.6，可与地图重合。
    """
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    features = ['totalCostInflated', 'Dam']
    feature_map = {
        'totalCostInflated': 'Inflated cost\n(log(1 + x) USD)',
        'Dam': 'Dams'
    }

    df_viz['ZCTA5CE20'] = df_viz['ZCTA5CE20'].astype(str)
    gdf_zcta['ZCTA5CE20'] = gdf_zcta['ZCTA5CE20'].astype(str)

    gdf_list = []
    for feat in features:
        gdf_merged = gdf_zcta.merge(df_viz[['ZCTA5CE20', feat]], on='ZCTA5CE20', how='left')
        gdf_list.append(gdf_merged)

    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 20

    fig, axes = plt.subplots(1, 2, figsize=(10, 6), gridspec_kw={'wspace': 0.02})

    for i, feat in enumerate(features):
        gdf = gdf_list[i]
        ax = axes[i]

        # 绘图并获取 color data
        cmap = plt.get_cmap('OrRd')
        vmin = gdf[feat].min()
        vmax = gdf[feat].max()
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

        gdf.plot(
            column=feat,
            cmap=cmap,
            linewidth=0.2,
            edgecolor='grey',
            ax=ax,
            norm=norm
        )

        # ax.set_title(feature_map[feat], fontsize=20)
        ax.axis('off')
        ax.text(0.5, -0.04, f"({chr(97 + i)})", transform=ax.transAxes,
                ha='center', va='top', fontsize=24)

        # 自定义浮动 colorbar（纵向）位置，重叠左下角
        cax = fig.add_axes([0.04 + i * 0.51, 0.18, 0.02, 0.5])  # [left, bottom, width, height]
        cb = mpl.colorbar.ColorbarBase(
            cax,
            cmap=cmap,
            norm=norm,
            orientation='vertical'
        )
        cb.set_label(feature_map[feat], fontsize=24)
        cb.ax.tick_params(labelsize=14)

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # 不留边
    plt.savefig("./imgs/plot_heatmap_2.pdf", format="pdf", bbox_inches='tight', pad_inches=0, dpi=300)
    plt.show()


if __name__ == '__main__':
    from load_data import load_zcta, load_processed_data
    from features import prepare_features

    df_claims, df_hydro, df_storms = load_processed_data()
    X, y, df_viz = prepare_features(df_claims, df_hydro, df_storms)
    gdf_zcta = load_zcta()
    plot_heatmap(df_viz, gdf_zcta, feature='Dam')
