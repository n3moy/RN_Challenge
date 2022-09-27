import os
import json
from typing import List
from tqdm import tqdm
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")


def reduce_mem_usage(
    data_in: pd.DataFrame,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Reduces pd.DataFrame memory usage based on columns types
    """
    numerics = ["int8", "int16", "int32", "int64", "float16", "float32", "float64"]
    start_mem = data_in.memory_usage().sum() / 1024 ** 2

    for col in data_in.columns:
        col_type = data_in[col].dtypes
        if col_type in numerics:
            c_min = data_in[col].min()
            c_max = data_in[col].max()
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    data_in[col] = data_in[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    data_in[col] = data_in[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    data_in[col] = data_in[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    data_in[col] = data_in[col].astype(np.int64)
            else:
                if (
                        c_min > np.finfo(np.float32).min
                        and c_max < np.finfo(np.float32).max
                ):
                    data_in[col] = data_in[col].astype(np.float32)
                else:
                    data_in[col] = data_in[col].astype(np.float64)
    end_mem = data_in.memory_usage().sum() / 1024 ** 2
    if verbose:
        print(
            "Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)".format(
                end_mem, 100 * (start_mem - end_mem) / start_mem
            )
        )
    return data_in


def read_cfg(cfg_path):
    with open(cfg_path, 'r') as fin:
        cfg = json.load(fin)
    return cfg


def filter_data_by_test(
    input_data: pd.DataFrame,
    test_cols_path: str
) -> pd.DataFrame:
    # COLS_PATH = "../configs/test_outer_nan_cols.json"
    test_nan_cols = read_cfg(test_cols_path)
    out_data = input_data.loc[:, ~input_data.columns.isin(test_nan_cols)]
    # This columns has leak and not used in test sample
    out_data = out_data.drop("CurrentTTF", axis=1)

    return out_data


def filter_numeric(
    input_data: pd.DataFrame,
) -> pd.DataFrame:
    return input_data.select_dtypes(include=[float, int])


def get_dynamic_features(
    description_path: str = "../../reports/Описание признаков.csv",
    dynamic_level: float = 2.2
) -> pd.DataFrame:
    description_file = pd.read_csv(description_path)
    descr_cols = description_file.columns
    # Long name
    col_to_filter = descr_cols[-1]
    # Only column with dynamic level = 1
    highly_dynamic_col = descr_cols[-2]

    description_file[col_to_filter] = description_file[col_to_filter].replace("na", np.nan)
    description_file[col_to_filter] = description_file[col_to_filter].fillna(3)
    description_file[col_to_filter] = description_file[col_to_filter].astype(np.float16)

    dynamic_features = description_file[
        (description_file[col_to_filter] <= dynamic_level) | (description_file[highly_dynamic_col] <= dynamic_level)]\
        ["Название столбца"].values.tolist()

    return dynamic_features


def filter_data(
    data_folder_path: str,
    output_path: str,
    dynamic_level: float,
) -> None:
    dtypes_path = "../../configs/column_dtypes.json"
    test_cols_path = "../../configs/test_outer_nan_cols.json"
    save_cols_path = "../../configs/save_dynamic_cols.json"
    # description_path = "../../reports/Описание признаков.csv"

    columns_dtypes = read_cfg(dtypes_path)
    fnames = os.listdir(data_folder_path)
    # Save them cause they are not classified by dynamic in description hence might be removed after filtering in step 1
    save_features = read_cfg(save_cols_path)
    dynamic_features = get_dynamic_features(dynamic_level=dynamic_level)
    print(f"DYNAMIC FEATURES amount {len(dynamic_features)}")
    # print(dynamic_features)
    print("Starting filtering data")
    for fname in tqdm(fnames):
        file_path = os.path.join(data_folder_path, fname)
        data_file = pd.read_parquet(file_path)

        for k, v in columns_dtypes.items():
            try:
                data_file[k] = data_file[k].astype(v)
            except KeyError:
                # print(f"{k} not found in file")
                continue

        save_vals = data_file[save_features]
        # Step 1 - Take dynamic features from features description with dynamic <= 'dynamic_level'
        dynamic_data = data_file.loc[:, dynamic_features]
        # print(f"COLUMNS AFTER DYNAMIC FILTER {dynamic_data.shape[1]}\n{dynamic_data.columns}")
        dynamic_data["SK_Calendar"] = pd.to_datetime(dynamic_data["SK_Calendar"])
        dynamic_data = dynamic_data.set_index("SK_Calendar")
        # Step 2 - Take columns that indeed present in tests samples
        test_filtered_data = filter_data_by_test(dynamic_data, test_cols_path)
        # print(f"COLUMNS AFTER TEST FILTER {test_filtered_data.shape[1]}\n{test_filtered_data.columns}")
        # Step 3 - Take only numeric columns, categorical features left for later use and another processing
        numeric_data = filter_numeric(test_filtered_data)
        # print(f"COLUMNS AFTER NUMERIC FILTER {numeric_data.shape[1]}\n{numeric_data.columns}")
        # OK lets take them back
        numeric_data[save_features] = save_vals

        save_path = os.path.join(output_path, fname)
        numeric_data.to_parquet(save_path)


def build_window_features(
    input_data: pd.DataFrame,
    target_columns: List[str]
):
    windows = [7, 14, 28]
    out_data = input_data.copy()

    for col in target_columns:
        for window in windows:
            out_data[f"{col}_mean_{window}"] = (out_data[col].rolling(min_periods=1, window=window).mean())
            out_data[f"{col}_std_{window}"] = (out_data[col].rolling(min_periods=1, window=window).std())
            out_data[f"{col}_max_{window}"] = (out_data[col].rolling(min_periods=1, window=window).max())
            out_data[f"{col}_min_{window}"] = (out_data[col].rolling(min_periods=1, window=window).min())
            out_data[f"{col}_spk_{window}"] = np.where(
                (out_data[f"{col}_mean_{window}"] == 0), 0, out_data[col] / out_data[f"{col}_mean_{window}"]
            )
        # out_data[f"{col}_deriv"] = pd.Series(np.gradient(out_data[col]), out_data.index)
        # out_data[f"{col}_squared"] = np.power(out_data[col], 2)
        # out_data[f"{col}_root"] = np.power(out_data[col], 0.5)

    return out_data


def build_features(
    data_path: str,
    output_path: str,
    dynamic_level: float,
    do_filter: bool,
    filtered_data_path: str = None
):
    if do_filter:
        filter_data(data_path, filtered_data_path, dynamic_level)
        fnames = os.listdir(filtered_data_path)
        calc_path = filtered_data_path
    else:
        fnames = os.listdir(data_path)
        calc_path = data_path

    # Columns in total 2657
    print("Starting calculation window features")
    for fname in tqdm(fnames[: 10]):
        file_path = os.path.join(calc_path, fname)
        data_file = pd.read_parquet(file_path)
        data_file["SK_Calendar"] = pd.to_datetime(data_file["SK_Calendar"])
        data_file = data_file.set_index("SK_Calendar")
        # data_file = reduce_mem_usage(data_file)
        window_columns = data_file.drop("daysToFailure", axis=1).columns.tolist()
        window_featured_file = build_window_features(data_file, window_columns)

        save_path = os.path.join(output_path, fname)
        window_featured_file.to_parquet(save_path)


if __name__ == "__main__":
    DATA_PATH = "../../data/processed"
    OUT_PATH = "../../data/featured"
    # 1 - постоянно изменяющийся (ежедневно),
    # 2.1 - периодически изменяющийся (допустимо изменение раз в неделю),
    # 2.2 - периодически изменяющийся (допустимо изменение раз в месяц),
    # 3.1 - статичный (обязательное изменение после останова / ремонта),
    # 3.2 - статичный (возможное изменение после останова / ремонта),
    # 3.3 - статичный (не изменяется)
    # Условие:    ... <= dynamic level
    DYNAMIC_LEVEL = 2.2
    DO_FILTER = True
    FILTERED_PATH = "../../data/filtered"
    build_features(DATA_PATH, OUT_PATH, DYNAMIC_LEVEL, DO_FILTER, FILTERED_PATH)

