import json
# import psutil
# import argparse

import pandas as pd
# from pandas.api.types import is_datetime64_any_dtype as is_datetime
import numpy as np
from typing import List

from functools import partial
from multiprocessing import Pool
import warnings
from pathlib import Path
from tqdm import tqdm
import joblib

# from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor

from tsfresh import extract_features
from tsfresh.feature_extraction import MinimalFCParameters
from sklearn.model_selection import train_test_split
import sklearn.metrics
from sklearn.preprocessing import OneHotEncoder

from scripts.features.dynamic_features import build_features, filter_data
from scripts.features.dynamic_features import build_window_features


def process_one_well(
        column_dtypes,
        well_path,
        save_features: List[str],
        dynamic_features: List[str],
        test_nan_features: List[str],
        window_features: List[str]
) -> pd.DataFrame:
    df = pd.read_csv(well_path, low_memory=False, dtype=column_dtypes)

    df['SK_Calendar'] = pd.to_datetime(df['SK_Calendar'], format='%Y-%m-%d')
    df = df.set_index("SK_Calendar")
    df['lastStartDate'] = pd.to_datetime(df['lastStartDate'], format='%Y-%m-%d')
    df['SKLayers'] = df['SKLayers'].fillna(value='').str.split(';').map(len)

    df = filter_data(
        df, save_features, dynamic_features, test_nan_features
    )

    df = build_features(df, window_features)

    df = df.fillna(method='ffill')
    df = df.fillna(method='bfill')
    df = df.fillna(value=-1)

    return df


def process_single_df(split, column_dtypes, input_features, window_features, well_path):
    df_in = pd.read_csv(well_path, low_memory=False, dtype=column_dtypes)
    # df = pd.read_parquet(well_path)
    df = df_in[input_features + ['SK_Well', 'SK_Calendar', 'lastStartDate']]

    # df[window_features] = df[window_features].fillna(method='ffill')
    # df[window_features] = df[window_features].fillna(method='bfill')
    # df[window_features] = df[window_features].fillna(value=-1)

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
    df[tsfresh_features] = df[tsfresh_features].fillna(method='ffill')
    df[tsfresh_features] = df[tsfresh_features].fillna(method='bfill')
    df[tsfresh_features] = df[tsfresh_features].fillna(value=-1)
    # df, window_cols = build_window_features(df, tsfresh_features)
    #
    window_cols = []
    # df[window_cols] = df[window_cols].fillna(method='ffill')
    # df[window_cols] = df[window_cols].fillna(method='bfill')
    # df[window_cols] = df[window_cols].fillna(value=-1)

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


def read_cfg(cfg_path):
    with open(cfg_path, 'r', encoding="utf-8") as fin:
        cfg = json.load(fin)
    return cfg


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


def train(
        data_dir: Path,
        num_workers: int,
        model_dir: Path
):
    cfg_dir = Path(__file__).parent.parent.parent / 'configs'
    column_dtypes = read_cfg(cfg_dir / 'column_dtypes.json')
    tsfresh_dict = read_cfg(cfg_dir / 'non_zero_lasso_cols_v2_cat.json')
    non_window_groups = ["Оборудование", "Отказы", "Скважинно-пластовые условия", "Расчетные параметры", "Категориальные"]
    input_features = []
    window_features = []

    for key in tsfresh_dict.keys():
        if key not in non_window_groups:
            window_features.extend(tsfresh_dict[key])
        input_features.extend(tsfresh_dict[key])
    # del input_features[["daysToFailure", "CurrentTTF"]]
    # input_features = list(set(input_features))
    train_df, _ = make_processed_df(data_dir, 'train', num_workers, column_dtypes, input_features, window_features)
    X = train_df.drop(columns=[
        'FailureDate', 'daysToFailure', 'CurrentTTF', 'SK_Calendar', 'lastStartDate', 'CalendarStart', "SK_Well"
    ])
    y = train_df['daysToFailure']
    cat_features = list(X.select_dtypes('object').columns)
    num_features = list(X.select_dtypes(np.number).columns)

    print(tsfresh_dict["Категориальные"] == cat_features)
    print(len(tsfresh_dict["Категориальные"]), len(cat_features))

    X[tsfresh_dict["Категориальные"]] = X[
        tsfresh_dict["Категориальные"]
    ].astype(str).fillna('')
    X[num_features] = X[num_features].fillna(method='ffill')
    X[num_features] = X[num_features].fillna(method='bfill')
    X[num_features] = X[num_features].fillna(value=-1)

    encoder = OneHotEncoder(handle_unknown="ignore", max_categories=7, sparse=False)
    transform_cols = pd.DataFrame(encoder.fit_transform(X[tsfresh_dict["Категориальные"]]),
                                  columns=encoder.get_feature_names_out())
    X = X.drop(columns=tsfresh_dict["Категориальные"])
    X = pd.concat([X, transform_cols], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(verbose=100, n_jobs=-1)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    rmsle = sklearn.metrics.mean_squared_log_error(y_test, pred, squared=False)
    print(rmsle)
    joblib.dump(model, model_dir / "model_rf_cat_v1.joblib")
    joblib.dump(encoder, model_dir / "ohe.joblib")
    # model.save_model("../../model/model_lasso.cbm", format='cbm')


def predict(
        data_dir: Path,
        num_workers: int,
        model_dir: Path
):
    cfg_dir = Path(__file__).parent.parent.parent / 'configs'
    column_dtypes = read_cfg(cfg_dir / 'column_dtypes.json')
    tsfresh_dict = read_cfg(cfg_dir / 'non_zero_lasso_cols_v2_cat.json')
    input_features = []
    non_window_groups = ["Оборудование", "Отказы", "Скважинно-пластовые условия", "Расчетные параметры", "Категориальные"]
    window_features = []

    for key in tsfresh_dict.keys():
        if key not in non_window_groups:
            window_features.extend(tsfresh_dict[key])
        input_features.extend(tsfresh_dict[key])

    # input_features = list(set(input_features))
    test_df, well_paths = make_processed_df(data_dir, 'test', num_workers, column_dtypes, input_features,
                                            window_features)
    test_df = test_df.drop(columns=[
        'SK_Calendar', 'lastStartDate', 'CalendarStart', 'SK_Well'
    ])
    # X = test_df.drop(columns=['FailureDate', 'daysToFailure', 'CurrentTTF', 'SK_Calendar', 'lastStartDate', 'CalendarStart'])

    cat_features = list(test_df.select_dtypes('object').columns)
    num_features = list(test_df.select_dtypes(np.number).columns)

    print(tsfresh_dict["Категориальные"] == cat_features)
    print(len(tsfresh_dict["Категориальные"]), len(cat_features))
    # , min_frequency=6000
    test_df[tsfresh_dict["Категориальные"]] = test_df[
        tsfresh_dict["Категориальные"]
    ].astype(str).fillna('')
    test_df[num_features] = test_df[num_features].fillna(method='ffill')
    test_df[num_features] = test_df[num_features].fillna(method='bfill')
    test_df[num_features] = test_df[num_features].fillna(value=-1)
    encoder = joblib.load(model_dir / "ohe.joblib")
    transform_cols = pd.DataFrame(encoder.transform(test_df[tsfresh_dict["Категориальные"]]),
                                  columns=encoder.get_feature_names_out())
    test_df = test_df.drop(columns=tsfresh_dict["Категориальные"])
    test_df = pd.concat([test_df, transform_cols], axis=1)

    print(test_df.shape)
    model = joblib.load(model_dir / "model_rf_v2_cat.joblib")
    preds = model.predict(test_df)
    sub = pd.DataFrame({'filename': [well_path.name for well_path in well_paths], 'daysToFailure': preds})
    sub.to_csv("../../data/sub.csv", index=False)
    return sub


if __name__ == "__main__":
    # cfg_dir = Path(__file__).parent.parent.parent / 'configs'
    # column_dtypes = read_cfg(cfg_dir / 'column_dtypes.json')
    # tsfresh_features = read_cfg(cfg_dir / 'tsfresh_features.json')
    # DATA_FILE = "../../data/processed/00a28f99.parquet"
    #
    # test_file = pd.read_parquet(DATA_FILE)
    # print(f"Shape before: {test_file.shape}")

    DO_TRAIN = False
    DO_TEST = True

    if DO_TRAIN:
        train_data_dir = Path(__file__).parent.parent.parent / "data" / "processed"
        num_workers = 4
        model_dir = Path(__file__).parent.parent.parent / "model"
        train(train_data_dir, num_workers, model_dir)
    if DO_TEST:
        test_data_dir = Path(__file__).parent.parent.parent / "data" / "test"
        num_workers = 4
        model_dir = Path(__file__).parent.parent.parent / "model"
        ans = predict(test_data_dir, num_workers, model_dir)

