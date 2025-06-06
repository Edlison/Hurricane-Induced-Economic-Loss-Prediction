import geopandas as gpd
import pandas as pd


def load_claims():
    df = pd.read_csv('./data/FimaNfipClaims.csv')
    # 筛选条件
    df_filtered = df[(df['state'] == 'FL') & (df['yearOfLoss'].between(1985, 2023))]  # select FL, 1985 - 2023
    df_filtered = df_filtered[
        df_filtered['buildingReplacementCost'].notna() &
        (df_filtered['buildingReplacementCost'] != 0)
        ]  # remove invalid
    valid_causes = ['1', '2', '4', 'A']
    df_filtered = df_filtered[df_filtered['causeOfDamage'].isin(valid_causes)]  # causeOfDamage
    # 打印总数
    print(f"Claim valid data: {len(df_filtered)}")  # causeOfDamage buildingReplacementCost latitude longitude

    # print(df.columns)
    # Select Flood: ratedFloodZone -> ratedFloodZoneMapped
    def map_flood_zone(zone):
        zone = str(zone)
        if 'V' in zone:
            return 'V'
        elif 'A' in zone:
            return 'A'
        elif any(x in zone for x in ['B', 'C', 'X']):
            return 'X'
        elif 'D' in zone:
            return 'D'
        else:
            return 'Unknown'  # 可选处理其他异常值

    df_filtered['ratedFloodZoneMapped'] = df_filtered['ratedFloodZone'].apply(map_flood_zone)
    df_filtered['floodZoneCurrentMapped'] = df_filtered['floodZoneCurrent'].apply(map_flood_zone)
    # floodCharacteristicsIndicator 20w NA
    # floodWaterDuration 20w NA
    # floodproofedIndicator inbalance [20w, 32]
    # cause = df_filtered[['latitude', 'longitude']]
    # print(cause.describe())
    # print(cause.value_counts(dropna=False))
    return df_filtered[
        ['buildingReplacementCost', 'ratedFloodZoneMapped', 'ratedFloodZoneMapped', 'latitude', 'longitude']]


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


def claims_by_zcta():
    df_claim = load_claims()
    df_zcta = load_zcta()
    # 步骤 1: 将 claim 转为 GeoDataFrame
    gdf_claim = gpd.GeoDataFrame(
        df_claim,
        geometry=gpd.points_from_xy(df_claim['longitude'], df_claim['latitude']),
        crs="EPSG:4326"  # 标准 WGS84 经纬度坐标系
    )
    # 步骤 2: 确保 ZCTA 使用相同 CRS
    gdf_zcta = df_zcta.to_crs('EPSG:4326')
    # 步骤 3: 空间连接，获得每个 claim 所在的 ZCTA 区域
    gdf_joined = gpd.sjoin(gdf_claim, gdf_zcta[['ZCTA5CE20', 'geometry']], how='inner', predicate='within')
    # 步骤 4: 按 ZCTA 分组并求和
    res = gdf_joined.groupby('ZCTA5CE20')['buildingReplacementCost'].sum().reset_index()
    res.rename(columns={'buildingReplacementCost': 'buildingCostSum'}, inplace=True)
    print(res.head(5))
    print(res.describe())


if __name__ == '__main__':
    claims_by_zcta()
