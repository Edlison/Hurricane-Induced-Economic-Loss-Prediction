import geopandas as gpd
import pandas as pd


def claims():
    df = pd.read_csv('./data/FimaNfipClaims.csv')
    print(df.head(5))
    cost = df['buildingReplacementCost']
    print(cost.describe())
    date = df['dateOfLoss']
    print(date.head(100))
    print(date.describe())


def storm():
    df = pd.read_csv('./data/ibtracs.ALL.list.v04r01.csv')
    print(df.head(5))
    sid = df['SID']
    print(sid.describe())
    date = df['ISO_TIME']
    print(date.describe())
    dist2land = df['DIST2LAND']
    print(dist2land.describe())


def load_hydro(verbose=False):
    df = pd.read_csv('./data/Florida_Hydrography_Dataset_(FHD)_(Formerly_NHD)_-_Point_Event_Feature_Class_(24k).csv')
    event = df['EVENTTYPE']
    coord = df[['X', 'Y']]
    hydro = df[['X', 'Y', 'EVENTTYPE']]
    if verbose:
        print(df.head(5))
        print(event.describe())
        print(coord.describe())
    return hydro


def load_zcta(verbose=False):
    # 读取 .shp 文件（GeoPandas 会自动识别同名的 .dbf, .shx, .prj 等文件）
    gdf = gpd.read_file("./data/tl_2024_us_zcta520/tl_2024_us_zcta520.shp")
    gdf = gdf[gdf['ZCTA5CE20'].notna()]
    gdf['ZCTA5CE20'] = gdf['ZCTA5CE20'].astype(int)
    florida_df = gdf[(gdf['ZCTA5CE20'] >= 32000) & (gdf['ZCTA5CE20'] <= 34999)]
    if verbose:
        print(florida_df.head())
        print(florida_df.columns)
        print(florida_df['ZCTA5CE20'].value_counts())
        print(florida_df['geometry'].value_counts())
    return florida_df[['ZCTA5CE20', 'geometry']]


def hydro_by_zcta():
    eventtype_dict = {
        1: "Gaging Station",
        2: "Dam",
        3: "Divergence Structure = General",
        4: "Divergence Structure = Withdrawing",
        5: "Divergence Structure = Contributing",
        57001: "Streamgage; Status=Active; Record=Continuous",
        57002: "Streamgage; Status=Active; Record=Partial",
        57003: "Streamgage; Status=Inactive",
        57004: "Water Quality Station",
        57100: "Dam",
        57201: "Flow Alteration=Addition",
        57202: "Flow Alteration=Removal",
        57203: "Flow Alteration Unknown",
        57300: "Hydrologic Unit Outlet",
        6: "Water Quality Station"
    }
    # 读取 hydrography 数据
    df_hydro = load_hydro()
    # 构造 geometry 点
    gdf_hydro = gpd.GeoDataFrame(
        df_hydro,
        geometry=gpd.points_from_xy(df_hydro['X'], df_hydro['Y']),
        crs="EPSG:3086"  # US National Atlas Equal Area, Albers
    )
    # 投影到 WGS84（经纬度），以便与 ZCTA 匹配
    gdf_hydro = gdf_hydro.to_crs("EPSG:4326")
    gdf_zcta = load_zcta()
    gdf_zcta = gdf_zcta.to_crs("EPSG:4326")
    # 使用空间连接，判断每个点是否在某个 ZCTA 多边形内
    joined = gpd.sjoin(gdf_hydro, gdf_zcta, how='left', predicate='within')
    counts_by_zcta = joined.groupby('ZCTA5CE20').size().reset_index(name='water_feature_count')
    joined['EventDescription'] = joined['EVENTTYPE'].map(eventtype_dict)
    existing_eventtypes = joined['EVENTTYPE'].dropna().unique()
    existing_eventtypes.sort()
    print(existing_eventtypes)

    # 映射 EVENTTYPE 到更简化的分类
    def simplify_eventtype(evt):
        if evt in [57001, 57002, 57003]:
            return 'Streamgage'
        elif evt == 57004:
            return 'Station'
        elif evt == 57100:
            return 'Dam'
        elif evt == 57300:
            return 'Outlet'
        else:
            return 'Other'

    # 添加简化后的分类列
    joined['EventCategory'] = joined['EVENTTYPE'].apply(simplify_eventtype)
    # 统计每个 ZCTA 内，各类 EventCategory 的数量
    event_counts_by_zcta = (
        joined.groupby(['ZCTA5CE20', 'EventCategory'])
        .size()
        .reset_index(name='count')
        .pivot(index='ZCTA5CE20', columns='EventCategory', values='count')
        .fillna(0)
        .astype(int)
        .reset_index()
    )
    # 可选：将列名映射为 eventtype_dict 中的名称（更可读）
    event_counts_by_zcta = event_counts_by_zcta.rename(columns=eventtype_dict)
    event_counts_by_zcta['ZCTA5CE20'] = event_counts_by_zcta['ZCTA5CE20'].astype(int)
    print(event_counts_by_zcta)
    print(event_counts_by_zcta.describe())


if __name__ == '__main__':
    hydro_by_zcta()
