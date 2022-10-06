import pandas as pd
from sklearn.preprocessing import OneHotEncoder

def process_cat_feauters_single_df(df, 
                                   cols = None, 
                                   mode = None, 
                                   handle_date = True,
                                   cat_features_names_file = None):
    if handle_date:
        df['SK_Calendar'] = pd.to_datetime(df['SK_Calendar'], format='%Y-%m-%d')
        df['lastStartDate'] = pd.to_datetime(df['lastStartDate'], format='%Y-%m-%d')
        if 'FailureDate' in df:
            df['FailureDate'] = pd.to_datetime(df['FailureDate'], format='%Y-%m-%d')
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
        encoder = OneHotEncoder(handle_unknown = 'ignore', sparse= False)
        if cat_features_names_file is not None:
            features_df = pd.read_csv(cat_features_names_file)[cols.drop('SK_Well')]
            encoder.fit(features_df)
        transform_cols = pd.DataFrame(encoder.transform(df[cols.drop('SK_Well')]),
                                      columns = encoder.get_feature_names_out())
        transform_cols.index = df.index
        df = df.drop(cols, axis=1)
        df = pd.concat([df, transform_cols], axis=1)
    return df, cols

if __name__ == '__main__':
    from pathlib import Path
    BASE_FOLDER = Path(__file__).parent.parent.parent
    WELL_PATH = BASE_FOLDER / 'data' / 'processed' / '000bb919.parquet'
    df = pd.read_parquet(WELL_PATH)
    CAT_FEATURES_NAMES_PATH = BASE_FOLDER / 'reports' / 'categorical_features_unique_vals.csv'
    processed_df = process_cat_feauters_single_df(df, 
                                   cols = None, 
                                   mode = 'one-hot', 
                                   handle_date = True,
                                   cat_features_names_file = CAT_FEATURES_NAMES_PATH)
    print(processed_df[0])