import argparse
import json
import warnings

import pandas as pd
import psutil

from functools import partial
from multiprocessing import Pool
from pathlib import Path

from catboost import CatBoostRegressor
from tsfresh import extract_features
from tsfresh.feature_extraction import MinimalFCParameters
from tqdm import tqdm


def process_single_df(split, column_dtypes, tsfresh_features, well_path):
    df = pd.read_csv(well_path, low_memory=True, dtype=column_dtypes)

    df['SK_Calendar'] = pd.to_datetime(df['SK_Calendar'], format='%Y-%m-%d')
    df['lastStartDate'] = pd.to_datetime(df['lastStartDate'], format='%Y-%m-%d')

    df = df.merge(
        df[['SK_Well', 'SK_Calendar']].groupby('SK_Well').min().rename(
            columns={'SK_Calendar': 'CalendarStart'}
        ),
        on='SK_Well', how='left'
    )

    layer_vars = ['SK_Cyclic', 'SKLayers']
    df[layer_vars] = df[layer_vars].fillna(method='ffill')
    df[layer_vars] = df[layer_vars].fillna(method='bfill')

    df['SKLayers'] = df['SKLayers'].fillna(value='').str.split(';').map(len)
    df['CalendarDays'] = (df['SK_Calendar'] - df['CalendarStart']).dt.days

    df[tsfresh_features] = df[tsfresh_features].fillna(method='ffill')
    df[tsfresh_features] = df[tsfresh_features].fillna(method='bfill')
    df[tsfresh_features] = df[tsfresh_features].fillna(value=-1)

    if split == 'train':
        X_list = []
        min_date = df['SK_Calendar'].min()
        max_date = df['SK_Calendar'].max()
        date_range = pd.date_range(start=min_date, end=max_date, freq='30D')
        
        for date in date_range:
            df_filtered = df[df['SK_Calendar'] <= date]
            start_date = df_filtered['lastStartDate'].max()
            df_filtered = df_filtered[df_filtered['lastStartDate'] == start_date]
            X = df_filtered[
                ['SK_Well', 'CalendarDays']
            ].groupby(by='SK_Well').max()
            
            X = X.merge(df_filtered, on=['SK_Well', 'CalendarDays'])
            
            if not len(X):
                continue
            elif X['SK_Calendar'].iloc[0] != date:
                break

            df_extracted_features = extract_features(
                df_filtered[['SK_Well', 'CalendarDays'] + tsfresh_features],
                default_fc_parameters=MinimalFCParameters(),
                column_id='SK_Well',
                column_sort='CalendarDays',
                disable_progressbar=True,
                n_jobs=1
            )
            df_extracted_features = df_extracted_features.reset_index().rename(
                columns={'index': 'SK_Well'}
            )
            X = X.merge(df_extracted_features, on='SK_Well')
            X_list.append(X)
        return pd.concat(X_list) if X_list else None
    
    elif split == 'test':
        start_date = df['lastStartDate'].max()
        if start_date is pd.NaT:
            start_date = df['SK_Calendar'].min()
            df['lastStartDate'] = start_date
        
        df_filtered = df[df['lastStartDate'] == start_date]
        X = df_filtered[
            ['SK_Well', 'CalendarDays']
        ].groupby(by='SK_Well').max()
        X = X.merge(df_filtered, on=['SK_Well', 'CalendarDays'])

        df_extracted_features = extract_features(
            df_filtered[['SK_Well', 'CalendarDays'] + tsfresh_features],
            default_fc_parameters=MinimalFCParameters(),
            column_id='SK_Well',
            column_sort='CalendarDays',
            disable_progressbar=True,
            n_jobs=1
        )
        df_extracted_features = df_extracted_features.reset_index().rename(
            columns={'index': 'SK_Well'}
        )
        X = X.merge(df_extracted_features, on='SK_Well')
        return X


def make_processed_df(data_dir, split, num_workers, column_dtypes, tsfresh_features):
    well_paths = sorted(data_dir.rglob('*.csv'))
    partial_process_single_df = partial(process_single_df, split, column_dtypes, tsfresh_features)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=pd.errors.PerformanceWarning)
        with Pool(num_workers) as p:
            well_df_list = list(
                tqdm(p.imap(partial_process_single_df, well_paths), total=len(well_paths)))

    well_df_list = [well_df for well_df in well_df_list if well_df is not None]
    df = pd.concat(well_df_list, ignore_index=True)
    if split == 'test':
        assert len(df) == len(well_paths)
    return df, well_paths


def read_cfg(cfg_path):
    with open(cfg_path, 'r') as fin:
        cfg = json.load(fin)
    return cfg


def train(args):
    cfg_dir = Path(__file__).parent / 'configs'
    column_dtypes = read_cfg(cfg_dir / 'column_dtypes.json')
    tsfresh_features = read_cfg(cfg_dir / 'tsfresh_features.json')
    
    train_df, _ = make_processed_df(args.data_dir, 'train', args.num_workers, column_dtypes, tsfresh_features)
    X_train = train_df.drop(columns=['CurrentMTTF', 'FailureDate', 'daysToFailure'])
    y_train = train_df['daysToFailure']
    cat_features = list(X_train.select_dtypes('object').columns)
    X_train[cat_features] = X_train[
        cat_features
    ].astype(str).fillna('')
    model = CatBoostRegressor(
        cat_features=cat_features
    )
    model.fit(X_train, y_train)
    model.save_model(args.model_dir / 'model.cbm', format='cbm')


def predict(args):
    cfg_dir = Path(__file__).parent / 'configs'
    column_dtypes = read_cfg(cfg_dir / 'column_dtypes.json')
    tsfresh_features = read_cfg(cfg_dir / 'tsfresh_features.json')
    
    test_df, well_paths = make_processed_df(args.data_dir, 'test', args.num_workers, column_dtypes, tsfresh_features)
    cat_features = list(test_df.select_dtypes('object').columns)
    test_df[cat_features] = test_df[
        cat_features
    ].astype(str).fillna('')
    model = CatBoostRegressor().load_model(args.model_dir / 'model.cbm')
    preds = model.predict(test_df)
    sub = pd.DataFrame({'filename': [well_path.name for well_path in well_paths], 'daysToFailure': preds})
    sub.to_csv(args.output_dir / 'sub.csv', index=False)


def main(args):
    if args.mode == 'train':
        train(args)
    if args.mode == 'predict':
        predict(args)


def get_args(args=None):
    parser = argparse.ArgumentParser(
        description='Бейзлайн для прогнозирования отказов УЭЦН.'
    )
    parser.add_argument(
        '--mode',
        type=str, required=True, choices=['train', 'predict'],
        help='Режим запуска'
    )
    parser.add_argument(
        '-d', '--data-dir',
        type=Path, required=True,
        help='Путь к папке с данными.'
    )
    parser.add_argument(
        '-m', '--model-dir',
        type=Path, required=True,
        help='Путь к папке, в которой будет сохраняться и откуда ' +
             'будет загружаться модель.'
    )
    parser.add_argument(
        '-o', '--output-dir',
        type=Path, default=None,
        help='Путь к папке для сохранения результатов.'
    )
    parser.add_argument(
        '-n', '--num-workers',
        type=int, default=psutil.cpu_count(logical=False),
        help='Number of worker processes to use.'
    )
    return parser.parse_args(args)


if __name__ == '__main__':
    main(get_args())
