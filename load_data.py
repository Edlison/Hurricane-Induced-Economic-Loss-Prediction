import geopandas as gpd
import pandas as pd
from scipy.spatial import KDTree


def load_claims(verbose=False):
    df = pd.read_csv('./data/FimaNfipClaims.csv')
    hpi = pd.read_csv('./data/FLSTHPI.csv')
    # 筛选条件
    df_filtered = df[(df['state'] == 'FL') & (df['yearOfLoss'].between(1985, 2023))]  # select FL, 1985 - 2023
    df_filtered = df_filtered[
        df_filtered['buildingReplacementCost'].notna() &
        (df_filtered['buildingReplacementCost'] != 0)]  # remove building invalid
    # df_filtered = df_filtered[
    #     df_filtered['contentsReplacementCost'].notna() &
    #     (df_filtered['contentsReplacementCost'] != 0)]  # remove content invalid
    df_filtered['originalConstructionDate'] = pd.to_datetime(
        df_filtered['originalConstructionDate'],
        errors='coerce'
    )
    valid_construction = df_filtered['originalConstructionDate'].notna() & \
                         (df_filtered['originalConstructionDate'] != '')
    df_filtered = df_filtered[valid_construction]
    df_filtered['constructionYear'] = pd.to_datetime(df_filtered['originalConstructionDate']).dt.year

    valid_causes = ['1', '2', '4', 'A']
    df_filtered = df_filtered[df_filtered['causeOfDamage'].isin(valid_causes)]  # causeOfDamage
    # 打印总数
    print(f"Claim valid data: {len(df_filtered)}")  # causeOfDamage buildingReplacementCost latitude longitude

    # inflate
    # hpi 有 'observation_date' 和 'FLSTHPI' 两列，先提取年份
    hpi['year'] = pd.to_datetime(hpi['observation_date']).dt.year

    # 按年份聚合为年度 HPI（如果已是年度，则跳过）
    hpi_yearly = hpi.groupby('year')['FLSTHPI'].mean().reset_index()

    # 获取最新年份的 HPI 作为基准（当前价格水平）
    latest_hpi = hpi_yearly['FLSTHPI'].max()  # 2025-01-01

    # 合并到 df，确保 df 中有 yearOfLoss
    # df_filtered['yearOfLoss'] = pd.to_datetime(df_filtered['yearOfLoss'], errors='coerce').dt.year
    df_filtered['yearOfLoss'] = df_filtered['yearOfLoss'].astype(int)
    df_filtered = df_filtered.merge(hpi_yearly, left_on='yearOfLoss', right_on='year', how='left')
    print(df_filtered[['yearOfLoss', 'year', 'FLSTHPI']].drop_duplicates().sort_values('yearOfLoss'))

    # 计算通胀调整因子
    df_filtered['inflateFactor'] = latest_hpi / df_filtered['FLSTHPI']
    print(df_filtered['inflateFactor'].describe())

    # 执行通胀调整
    df_filtered['buildingReplacementCost_inflated'] = df_filtered['buildingReplacementCost'] * df_filtered[
        'inflateFactor']
    df_filtered['contentsReplacementCost_inflated'] = df_filtered['contentsReplacementCost'] * df_filtered[
        'inflateFactor']

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

    if verbose:
        # print(df_filtered['buildingReplacementCost_inflated'])
        print(df_filtered['buildingReplacementCost_inflated'].describe())
        # print(df_filtered['contentsReplacementCost_inflated'])
        print(df_filtered['contentsReplacementCost_inflated'].describe())

    return df_filtered[
        ['buildingReplacementCost', 'contentsReplacementCost', 'buildingReplacementCost_inflated',
         'contentsReplacementCost_inflated',
         'numberOfFloorsInTheInsuredBuilding', 'lowestFloorElevation', 'lowestAdjacentGrade', 'elevationDifference',
         'elevatedBuildingIndicator', 'ratedFloodZoneMapped', 'ratedFloodZoneMapped', 'occupancyType',
         'constructionYear', 'latitude', 'longitude']]


def load_storms(verbose=False):
    # 读取数据
    df = pd.read_csv('./data/ibtracs.ALL.list.v04r01.csv', low_memory=False)

    # 确保时间列是 datetime 格式
    df['ISO_TIME'] = pd.to_datetime(df['ISO_TIME'], errors='coerce')

    # 过滤 SubBasin 为 GM（Gulf of Mexico），并且 ISO_TIME 在 1985 到 2023 年之间
    mask = (df['SUBBASIN'] == 'GM') & \
           (df['ISO_TIME'].dt.year >= 1985) & (df['ISO_TIME'].dt.year <= 2023)
    df_filtered = df[mask]

    # 筛选所需列
    columns_of_interest = ['NAME', 'ISO_TIME', 'USA_WIND', 'USA_SSHS', 'USA_PRES', 'LON', 'LAT']
    df_filtered = df_filtered[columns_of_interest]

    if verbose:
        # 打印筛选结果
        print(f"Total filtered (GM, Date): {len(df_filtered)}")
        print(f"Total unique hurricane name: {df_filtered['NAME'].nunique()}")

    gdf_florida = load_florida()
    gdf_storm = gpd.GeoDataFrame(
        df_filtered,
        geometry=gpd.points_from_xy(df_filtered['LON'], df_filtered['LAT']),
        crs="EPSG:4326"
    )

    storm_columns = gdf_storm.columns.tolist()

    gdf_storm_in_florida = gpd.sjoin(gdf_storm, gdf_florida, how='inner', predicate='intersects')
    gdf_storm_in_florida['NAME'] = gdf_storm_in_florida['NAME_left']
    gdf_storm_cleaned = gdf_storm_in_florida[storm_columns]

    storm_names_in_florida = gdf_storm_in_florida['NAME_left'].dropna().unique()

    if verbose:
        print(f"Total data in Florida: {len(gdf_storm_cleaned)}")
        print(f"Total unique hurricane name: {len(storm_names_in_florida)}")
        print(gdf_storm_cleaned.head(5))
        print(gdf_storm_cleaned.columns)

    # str to numeric
    gdf_storm_cleaned['USA_WIND'] = pd.to_numeric(gdf_storm_cleaned['USA_WIND'], errors='coerce').fillna(0)
    gdf_storm_cleaned['USA_SSHS'] = pd.to_numeric(gdf_storm_cleaned['USA_SSHS'], errors='coerce').fillna(0)
    gdf_storm_cleaned['USA_PRES'] = pd.to_numeric(gdf_storm_cleaned['USA_PRES'], errors='coerce').fillna(0)

    return gdf_storm_cleaned


def load_florida():
    # 读取 US states 边界数据（例如来自 TIGER shapefile）
    gdf_states = gpd.read_file("./data/cb_2018_us_state_20m/cb_2018_us_state_20m.shp")  # 或你的 states shapefile 路径
    gdf_florida = gdf_states[gdf_states['NAME'] == 'Florida'].to_crs(epsg=4326)
    return gdf_florida


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
    print('Existing EventTypes: ', existing_eventtypes)

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
    event_counts_by_zcta.columns.name = None
    print(event_counts_by_zcta)
    print(event_counts_by_zcta.describe())
    return event_counts_by_zcta


def claims_by_zcta():
    df_claim = load_claims()
    df_zcta = load_zcta()

    # 步骤 1: 将 claim 转为 GeoDataFrame
    gdf_claim = gpd.GeoDataFrame(
        df_claim,
        geometry=gpd.points_from_xy(df_claim['longitude'], df_claim['latitude']),
        crs="EPSG:4326"
    )

    # 步骤 2: 确保 ZCTA 使用相同 CRS
    gdf_zcta = df_zcta.to_crs('EPSG:4326')

    # 步骤 3: 空间连接，获得每个 claim 所在的 ZCTA 区域
    gdf_joined = gpd.sjoin(
        gdf_claim,
        gdf_zcta[['ZCTA5CE20', 'geometry']],
        how='inner',
        predicate='within'
    )

    # 步骤 4: 聚合计算
    df_grouped = gdf_joined.groupby('ZCTA5CE20').agg({
        'buildingReplacementCost': 'sum',  # 房屋赔付金额
        'contentsReplacementCost': 'sum',  # 室内损失金额
        'buildingReplacementCost_inflated': 'sum',  # 房屋赔付金额Inflated
        'contentsReplacementCost_inflated': 'sum',  # 室内损失金额Inflated
        'lowestFloorElevation': 'mean',  # 平均最低楼层高度
        'lowestAdjacentGrade': 'mean',  # 平均最低比邻高度
        'elevationDifference': 'mean',  # 平均最低比邻高度
        'elevatedBuildingIndicator': 'sum',  # 架高房屋个数
        'numberOfFloorsInTheInsuredBuilding': 'mean',  # 平均楼层
        'constructionYear': 'mean',  # 平均建造年份
        'occupancyType': lambda x: x.mode().iloc[0] if not x.mode().empty else None,  # 占用类型众数
        'geometry': 'count'  # 行数 ≈ 理赔记录数量
    }).reset_index()

    # 重命名列
    df_grouped.rename(columns={
        'buildingReplacementCost': 'buildingCostSum',
        'contentsReplacementCost': 'contentCostSum',
        'buildingReplacementCost_inflated': 'buildingCostSumInflated',
        'contentsReplacementCost_inflated': 'contentsCostSumInflated',
        'numberOfFloorsInTheInsuredBuilding': 'numFloors',
        'lowestFloorElevation': 'lowestFloorElevation',
        'lowestAdjacentGrade': 'lowestAdjacentGrade',
        'elevationDifference': 'elevationDifference',
        'elevatedBuildingIndicator': 'elevatedBuildingIndicator',
        'constructionYear': 'avgConstructionYear',
        'occupancyType': 'mainOccupancyType',
        'geometry': 'claimCount'
    }, inplace=True)

    df_grouped['totalCost'] = df_grouped['buildingCostSum'] + df_grouped['contentCostSum']
    df_grouped['totalCostInflated'] = df_grouped['buildingCostSumInflated'] + df_grouped['contentsCostSumInflated']

    print(df_grouped)
    print(df_grouped.describe(include='all'))

    return df_grouped


def storms_by_zcta():
    # 构造 geometry 点
    gdf_storms = load_storms()
    df_zcta = load_zcta()
    # 步骤 2: 确保 ZCTA 使用相同 CRS
    gdf_zcta = df_zcta.to_crs('EPSG:4326')
    # 步骤 3: 空间连接，获得每个 claim 所在的 ZCTA 区域
    gdf_joined = gpd.sjoin(gdf_storms, gdf_zcta[['ZCTA5CE20', 'geometry']], how='inner', predicate='within')
    # 步骤 4: 按 ZCTA 分组并求和
    gdf_joined_sorted = gdf_joined[['ZCTA5CE20', 'NAME', 'USA_WIND', 'USA_SSHS', 'USA_PRES']] \
        .sort_values(by='ZCTA5CE20') \
        .reset_index(drop=True)
    print(gdf_joined_sorted)
    print(gdf_joined_sorted.describe())
    return gdf_joined_sorted


def save_data():
    import os
    # 创建目录（如果不存在）
    os.makedirs('./processed_data', exist_ok=True)

    # 加载数据
    claims = claims_by_zcta()
    hydro = hydro_by_zcta()
    storms = storms_by_zcta()

    # 去掉 geometry 列转为 DataFrame 并保存为 CSV
    claims.drop(columns='geometry', errors='ignore').to_csv('./processed_data/claims_by_zcta.csv', index=False)
    hydro.drop(columns='geometry', errors='ignore').to_csv('./processed_data/hydro_by_zcta.csv', index=False)
    storms.drop(columns='geometry', errors='ignore').to_csv('./processed_data/storms_by_zcta.csv', index=False)


def load_processed_data():
    df_zcta = load_zcta()
    # target
    df_claims = pd.read_csv('./processed_data/claims_by_zcta.csv')
    # need drop
    df_hydro = pd.read_csv('./processed_data/hydro_by_zcta.csv')
    df_hydro = filter_and_fill_hydro_in_claims(df_claims, df_hydro, df_zcta)
    # need fill
    df_storms = pd.read_csv('./processed_data/storms_by_zcta.csv')
    df_storms = fill_storm_by_geodistance(df_claims, df_storms, df_zcta)
    return df_claims, df_hydro, df_storms


def filter_and_fill_hydro_in_claims(df_claims, df_hydro, gdf_zcta):
    # 保留 claims 中涉及的 ZCTA
    target_zctas = set(df_claims['ZCTA5CE20'])
    df_hydro_filtered = df_hydro[df_hydro['ZCTA5CE20'].isin(target_zctas)].copy()

    # 找出缺失的 ZCTA
    existing_zctas = set(df_hydro_filtered['ZCTA5CE20'])
    zcta_missing = list(target_zctas - existing_zctas)

    # 准备 ZCTA 中心点坐标
    # 转换到适当的投影坐标系（比如等面积投影 EPSG:3086，美国地区常用）
    gdf_zcta_proj = gdf_zcta.to_crs(epsg=3086)

    # 计算几何中心
    gdf_zcta_proj['centroid'] = gdf_zcta_proj.geometry.centroid

    # 如果你最终还是希望保留经纬度坐标的中心点，需要再转换回WGS84
    gdf_zcta_centroid = gdf_zcta_proj.set_geometry('centroid').to_crs(epsg=4326)

    # 提取中心点经纬度
    gdf_zcta['lon'] = gdf_zcta_centroid.geometry.x
    gdf_zcta['lat'] = gdf_zcta_centroid.geometry.y

    # 添加坐标到 hydro 中
    df_hydro_coords = df_hydro.merge(
        gdf_zcta[['ZCTA5CE20', 'lon', 'lat']],
        on='ZCTA5CE20',
        how='left'
    )

    # 找出要填补的坐标
    missing_coords = gdf_zcta[gdf_zcta['ZCTA5CE20'].isin(zcta_missing)][['ZCTA5CE20', 'lon', 'lat']]

    # 建立 KDTree 用于最近邻查询
    tree = KDTree(df_hydro_coords[['lon', 'lat']].values)
    _, idx = tree.query(missing_coords[['lon', 'lat']].values)

    # 查找最近邻 hydro 特征并覆盖 ZCTA
    df_filled = df_hydro_coords.iloc[idx].copy().reset_index(drop=True)
    df_filled['ZCTA5CE20'] = missing_coords['ZCTA5CE20'].values

    # 只保留 hydro 特征列和 ZCTA
    hydro_cols = [col for col in df_hydro.columns if col != 'ZCTA5CE20']
    df_filled = df_filled[['ZCTA5CE20'] + hydro_cols]

    # 合并原始和补齐后的 hydro 数据
    df_hydro_filled = pd.concat([
        df_hydro_filtered,
        df_filled
    ]).drop_duplicates('ZCTA5CE20').reset_index(drop=True)

    # 最终只保留 target_zctas 范围
    df_hydro_filled = df_hydro_filled[df_hydro_filled['ZCTA5CE20'].isin(target_zctas)]

    return df_hydro_filled


def fill_storm_by_geodistance(df_claim, df_storm, gdf_zcta):
    # 转换到适当的投影坐标系（比如等面积投影 EPSG:3086，美国地区常用）
    gdf_zcta_proj = gdf_zcta.to_crs(epsg=3086)

    # 计算几何中心
    gdf_zcta_proj['centroid'] = gdf_zcta_proj.geometry.centroid

    # 如果你最终还是希望保留经纬度坐标的中心点，需要再转换回WGS84
    gdf_zcta_centroid = gdf_zcta_proj.set_geometry('centroid').to_crs(epsg=4326)

    # 提取中心点经纬度
    gdf_zcta['lon'] = gdf_zcta_centroid.geometry.x
    gdf_zcta['lat'] = gdf_zcta_centroid.geometry.y

    # 将经纬度信息添加到 storm ZCTA 上
    storm_with_coords = df_storm.merge(
        gdf_zcta[['ZCTA5CE20', 'lon', 'lat']],
        on='ZCTA5CE20',
        how='left'
    )

    # 找出在 claim 中但不在 storm 中的 ZCTA
    zcta_all = set(df_claim['ZCTA5CE20'])
    zcta_with_storm = set(storm_with_coords['ZCTA5CE20'])
    zcta_missing = zcta_all - zcta_with_storm

    # 提取缺失 ZCTA 的坐标
    missing_coords = gdf_zcta[gdf_zcta['ZCTA5CE20'].isin(zcta_missing)][['ZCTA5CE20', 'lon', 'lat']]

    # 使用 KDTree 查找最近邻
    tree = KDTree(storm_with_coords[['lon', 'lat']].values)
    _, idx = tree.query(missing_coords[['lon', 'lat']].values)

    # 用最近邻的 storm 数据补齐
    df_filled = storm_with_coords.iloc[idx].reset_index(drop=True)
    df_filled['ZCTA5CE20'] = missing_coords['ZCTA5CE20'].values  # 覆盖成目标 ZCTA

    # 合并原始和补齐部分
    df_storm_filled = pd.concat([
        df_storm[['ZCTA5CE20', 'USA_WIND', 'USA_SSHS', 'USA_PRES']],
        df_filled[['ZCTA5CE20', 'USA_WIND', 'USA_SSHS', 'USA_PRES']]
    ]).drop_duplicates('ZCTA5CE20')

    df_storm_filled = df_storm_filled[df_storm_filled['ZCTA5CE20'].isin(zcta_all)]

    return df_storm_filled.sort_values('ZCTA5CE20').reset_index(drop=True)


if __name__ == '__main__':
    # claims_by_zcta()
    # hydro_by_zcta()
    # storms_by_zcta()
    # load_claims(True)
    save_data()
    # df_claims, df_hydro, df_storms = load_processed_data()
    # print(df_claims)
    # print(df_claims.shape)
    # print(df_hydro)
    # print(df_hydro.shape)
    # print(df_storms)
    # print(df_storms.shape)
