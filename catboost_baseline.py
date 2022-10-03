import argparse
import json
import warnings
from typing import List

import numpy as np
import pandas as pd
import psutil

from functools import partial
from multiprocessing import Pool
from pathlib import Path
import joblib

# from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from tsfresh import extract_features
from tsfresh.feature_extraction import MinimalFCParameters
from tqdm import tqdm


def build_window_features(
    input_data: pd.DataFrame,
    target_columns: List[str]
):
    windows = [3, 7, 14]
    out_data = input_data.copy()
    out_cols = []
    for col in target_columns:
        for window in windows:
            out_data[f"{col}_mean_{window}"] = (out_data[col].rolling(min_periods=1, window=window).mean())
            out_data[f"{col}_std_{window}"] = (out_data[col].rolling(min_periods=1, window=window).std())
            out_data[f"{col}_max_{window}"] = (out_data[col].rolling(min_periods=1, window=window).max())
            out_data[f"{col}_min_{window}"] = (out_data[col].rolling(min_periods=1, window=window).min())
            out_data[f"{col}_spk_{window}"] = np.where(
                (out_data[f"{col}_mean_{window}"] == 0), 0, out_data[col] / out_data[f"{col}_mean_{window}"]
            )
            out_cols.extend(
                [f"{col}_mean_{window}",
                f"{col}_std_{window}",
                f"{col}_max_{window}",
                f"{col}_min_{window}",
                f"{col}_spk_{window}"]
            )
        # out_data[f"{col}_deriv"] = pd.Series(np.gradient(out_data[col]), out_data.index)
        # out_data[f"{col}_squared"] = np.power(out_data[col], 2)
        # out_data[f"{col}_root"] = np.power(out_data[col], 0.5)

    return out_data, out_cols


def process_single_df(split, column_dtypes, input_features, window_features, well_path):
    df_in = pd.read_csv(well_path, low_memory=False, dtype=column_dtypes)
    # df = pd.read_parquet(well_path)
    df = df_in[input_features + ['SK_Well', 'SK_Calendar', 'lastStartDate']]

    df[input_features] = df[input_features].fillna(method='ffill')
    df[input_features] = df[input_features].fillna(method='bfill')
    df[input_features] = df[input_features].fillna(value=-1)

    if split == "train":
        df[['FailureDate', 'daysToFailure', 'CurrentTTF']] = df_in[['FailureDate', 'daysToFailure', 'CurrentTTF']]

    df['SK_Calendar'] = pd.to_datetime(df['SK_Calendar'], format='%Y-%m-%d')
    df['lastStartDate'] = pd.to_datetime(df['lastStartDate'], format='%Y-%m-%d')

    df = df.merge(
        df[['SK_Well', 'SK_Calendar']].groupby('SK_Well').min().rename(
            columns={'SK_Calendar': 'CalendarStart'}
        ),
        on='SK_Well', how='left'
    )

    # df["SKLayers"] = df["SKLayers"].astype(str)
    # df['SKLayers'] = df['SKLayers'].fillna(value='').str.split(';').map(len)
    df['CalendarDays'] = (df['SK_Calendar'] - df['CalendarStart']).dt.days

    # tsfresh_features = df.select_dtypes(include=np.number).columns.tolist()
    tsfresh_features = window_features
    # df, window_cols = build_window_features(df, window_features)
    window_cols = []
    # df[window_cols] = df[window_cols].fillna(method='ffill')
    # df[window_cols] = df[window_cols].fillna(method='bfill')
    # df[window_cols] = df[window_cols].fillna(value=-1)
    # print(len(['SK_Well', 'CalendarDays'] + tsfresh_features + window_cols))
    # print(len(set(['SK_Well', 'CalendarDays'] + tsfresh_features + window_cols)))
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
                df_filtered[['SK_Well', 'CalendarDays'] + tsfresh_features + window_cols],
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
            df_filtered[['SK_Well', 'CalendarDays'] + tsfresh_features + window_cols],
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


def make_processed_df(data_dir, split, num_workers, column_dtypes, tsfresh_features, window_features):
    well_paths = sorted(data_dir.rglob('*.csv'))
    partial_process_single_df = partial(process_single_df, split, column_dtypes, tsfresh_features, window_features)

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
    tsfresh_dict = read_cfg(cfg_dir / 'non_zero_lasso_cols.json')
    non_window_groups = ["Оборудование", "Отказы", "Скважинно-пластовые условия"]
    input_features = []
    window_features = []

    for key in tsfresh_dict.keys():
        if key not in non_window_groups:
            window_features.extend(tsfresh_dict[key])
        input_features.extend(tsfresh_dict[key])

    train_df, _ = make_processed_df(args.data_dir, 'train', args.num_workers, column_dtypes, input_features, window_features)
    X_train = train_df.drop(columns=['FailureDate', 'daysToFailure', 'CurrentTTF', 'SK_Calendar', 'lastStartDate', 'CalendarStart'])
    y_train = train_df['daysToFailure']
    cat_features = list(X_train.select_dtypes('object').columns)
    # X_train[cat_features] = X_train[
    #     cat_features
    # ].astype(str).fillna('')
    X_train = X_train.drop(cat_features, axis=1)
    # model = CatBoostRegressor(
    #     cat_features=cat_features
    # )
    model = RandomForestRegressor(verbose=100, n_jobs=-1)
    model.fit(X_train, y_train)
    joblib.dump(args.model_dir / 'model.joblib')


def predict(args):
    cfg_dir = Path(__file__).parent / 'configs'
    column_dtypes = read_cfg(cfg_dir / 'column_dtypes.json')
    tsfresh_dict = read_cfg(cfg_dir / 'non_zero_lasso_cols.json')
    input_features = []
    non_window_groups = ["Оборудование", "Отказы", "Скважинно-пластовые условия"]
    window_features = []

    for key in tsfresh_dict.keys():
        if key not in non_window_groups:
            window_features.extend(tsfresh_dict[key])
        input_features.extend(tsfresh_dict[key])

    test_df, well_paths = make_processed_df(args.data_dir, 'test', args.num_workers, column_dtypes, input_features, window_features)
    test_df = test_df.drop(columns=['SK_Calendar', 'lastStartDate', 'CalendarStart'])
    cat_features = list(test_df.select_dtypes('object').columns)
    test_df = test_df.drop(cat_features, axis=1)
    # test_df[cat_features] = test_df[
    #     cat_features
    # ].astype(str).fillna('')
    # model = CatBoostRegressor().load_model(args.model_dir / 'model.cbm')
    model = joblib.load(args.model_dir / 'model.joblib')
    preds = model.predict(test_df)
    sub = pd.DataFrame({'filename': [well_path.name for well_path in well_paths], 'daysToFailure': preds})
    sub.to_csv(args.output_dir / 'submission.csv', index=False)


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
