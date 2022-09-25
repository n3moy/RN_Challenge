import pandas as pd


def process_cat_feauters_single_df(df, cols = None, mode = None, handle_date = True):
    if handle_date:
        df['SK_Calendar'] = pd.to_datetime(df['SK_Calendar'], format='%Y-%m-%d')
        df['lastStartDate'] = pd.to_datetime(df['lastStartDate'], format='%Y-%m-%d')
        df = df.merge(
        df[['SK_Well', 'SK_Calendar']].groupby('SK_Well').min().rename(
            columns={'SK_Calendar': 'CalendarStart'}
        ),
        on='SK_Well', how='left'
        )

        df['CalendarDays'] = (df['SK_Calendar'] - df['CalendarStart']).dt.days
        
    if 'SKLayers' in df:
        # Мб можно придумать способ лучше
        df['SKLayers'] = df['SKLayers'].fillna(value='').str.split(';').map(len)
        
    if cols is None:
        cols = df.select_dtypes(include=['object']).columns
        
    if mode is None:
        df[cols] = df[cols].astype(str).replace('None','')
    elif mode == 'one-hot':
        df = pd.get_dummies(df, columns = cols)
        
    return df, cols