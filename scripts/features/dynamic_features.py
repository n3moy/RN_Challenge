import json
from typing import List
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


def get_dynamic_features(
    description_path: str = "../../reports/Описание признаков.csv",
    dynamic_level: float = 2.2
) -> List[str]:
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
    input_data: pd.DataFrame,
    save_features: List[str],
    dynamic_features: List[str],
    test_nan_features: List[str]
) -> pd.DataFrame:
    filter_file = input_data.copy()

    save_vals = filter_file[save_features]

    # Step 1 - Take dynamic features from features description with dynamic <= 'dynamic_level'
    dynamic_data = filter_file.loc[:, dynamic_features]
    dynamic_data["SK_Calendar"] = pd.to_datetime(filter_file["SK_Calendar"])
    dynamic_data = dynamic_data.set_index("SK_Calendar")

    # Step 2 - Take columns that indeed present in tests samples
    test_filtered_data = dynamic_data.loc[:, ~dynamic_data.columns.isin(test_nan_features)]

    # Step 3 - Take only numeric columns, categorical features left for later use and another processing
    numeric_data = test_filtered_data.select_dtypes(include=[float, int])

    # Let's take them back
    numeric_data[save_features] = save_vals

    return numeric_data


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
    input_data: pd.DataFrame,
    window_columns: List[str] = None
) -> pd.DataFrame:
    data_file = input_data.copy()
    # data_file = reduce_mem_usage(data_file)
    if window_columns is None:
        window_columns = [col for col in data_file.columns if "failure" not in col.lower()]
        # window_columns = data_file.drop("daysToFailure", axis=1).columns.tolist()
    window_featured_file = build_window_features(data_file, window_columns)

    return window_featured_file


if __name__ == "__main__":
    # FOR TESTING PURPOSES
    DATA_FILE = pd.read_parquet("../../data/processed/00d89d23.parquet")
    # 1 - постоянно изменяющийся (ежедневно),
    # 2.1 - периодически изменяющийся (допустимо изменение раз в неделю),
    # 2.2 - периодически изменяющийся (допустимо изменение раз в месяц),
    # 3.1 - статичный (обязательное изменение после останова / ремонта),
    # 3.2 - статичный (возможное изменение после останова / ремонта),
    # 3.3 - статичный (не изменяется)
    # Условие:    ... <= dynamic level
    DYNAMIC_LEVEL = 3.1
    save_features = read_cfg("../../configs/save_dynamic_cols.json")
    dynamic_features = get_dynamic_features(dynamic_level=DYNAMIC_LEVEL)
    test_nan_features = read_cfg("../../configs/test_outer_nan_cols.json")
    filtered_data = filter_data(DATA_FILE, save_features, dynamic_features, test_nan_features)
    featured = build_features(filtered_data)
    n_rows = featured.shape[0]
    count = 0
    for col in featured.columns:
        nans = featured[col].isna().sum()
        if nans == n_rows:
            count += 1
    print(featured.shape)
    print(f"Fully nan cols: {count}")

