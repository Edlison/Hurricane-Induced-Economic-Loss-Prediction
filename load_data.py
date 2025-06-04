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


if __name__ == '__main__':
    storm()
