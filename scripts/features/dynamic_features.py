import os
import json

import pandas as pd
import numpy as np


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

    return out_data


def filter_numeric(
    input_data: pd.DataFrame,
) -> pd.DataFrame:
    return input_data.select_dtypes(include=[float, int])


def filter_data(
    data_folder_path: str,
    output_path: str,
    dynamic_level: int
) -> None:
    dtypes_path = "../../configs/column_dtypes.json"
    test_cols_path = "../../configs/test_outer_nan_cols.json"
    description_path = "../../reports/Описание признаков.csv"

    columns_dtypes = read_cfg(dtypes_path)
    fnames = os.listdir(data_folder_path)
    description_file = pd.read_csv(description_path)
    descr_cols = description_file.columns
    # Очень длинное название, пускай остается в таком виде
    col_to_filter = descr_cols[-1]
    dynamic_features = description_file[description_file[col_to_filter] <= dynamic_level]["Название столбца"].values.tolist()

    for fname in fnames:
        file_path = os.path.join(data_folder_path, fname)
        data_file = pd.read_parquet(file_path, low_memory=False, dtype=columns_dtypes)
        # Saving target so I dont lose it
        target = data_file["daysToFailure"]
        dynamic_data = data_file.loc[:, dynamic_features]
        test_filtered_data = filter_data_by_test(dynamic_data, test_cols_path)
        numeric_data = filter_numeric(test_filtered_data)
        numeric_data["daysToFailure"] = target

        save_path = os.path.join(output_path, fname)
        numeric_data.to_parquet(save_path, index=False)


def build_features(
    input_path: str,
    output_path: str
):
    fnames = os.listdir(input_path)

    for filename in fnames:
        file_path = os.path.join(input_path, filename)
        data_file = pd.read_csv(file_path, parse_dates=["time"])
        data_file = data_file.set_index("time")
        data_file = reduce_mem_usage(data_file)

        # Current and voltage unbalance
        voltage_names = ["voltageAB", "voltageBC", "voltageCA"]
        current_names = ["op_current1", "op_current2", "op_current3"]
        voltages = data_file[voltage_names]
        currents = data_file[current_names]
        mean_voltage = voltages.mean(axis=1)
        mean_current = currents.mean(axis=1)
        deviation_voltage = voltages.sub(mean_voltage, axis=0).abs()
        deviation_current = currents.sub(mean_current, axis=0).abs()

        data_file["voltage_unbalance"] = (
                deviation_voltage.max(axis=1).div(mean_voltage, axis=0) * 100
        )
        data_file["current_unbalance"] = (
                deviation_current.max(axis=1).div(mean_current, axis=0) * 100
        )

        # Impute zeros where currents are zeros
        data_file["current_unbalance"] = data_file["current_unbalance"].fillna(0)
        # Impute zeros where voltages are zeros
        data_file["voltage_unbalance"] = data_file["voltage_unbalance"].fillna(0)
        # I don't need currents anymore cause active power present variability
        # Lets keep only one voltage and current to save variability
        data_file["voltage"] = data_file["voltageAB"]
        data_file["current"] = data_file["op_current1"]
        data_file["resistance"] = np.where(
            (data_file["current"] == 0), 0, data_file["voltage"].div(data_file["current"], axis=0)
        )

        # Testing all ideas to choose best ones
        data_file["power_A"] = data_file["op_current1"] * data_file["voltageAB"] / 1000
        data_file["power_B"] = data_file["op_current2"] * data_file["voltageBC"] / 1000
        data_file["power_C"] = data_file["op_current3"] * data_file["voltageCA"] / 1000
        data_file["theory_power"] = data_file["power_A"] + data_file["power_B"] + data_file["power_C"]
        data_file["power_diff"] = data_file["active_power"] - data_file["theory_power"]

        data_file["power_lossesA"] = np.power(data_file["op_current1"], 2) * data_file["resistance"]
        data_file["power_lossesB"] = np.power(data_file["op_current2"], 2) * data_file["resistance"]
        data_file["power_lossesC"] = np.power(data_file["op_current3"], 2) * data_file["resistance"]

        # data_file["watercut"] = 1 - data_file["oil_rate"] / data_file["liquid_rate"]
        data_file["pressure_drop"] = data_file["intake_pressure"] - data_file["line_pressure"]
        data_file["theory_rate"] = data_file["pressure_drop"] * 8
        data_file["rate_diff"] = data_file["liquid_rate"] - data_file["theory_rate"]

        data_file["freq_ratio"] = data_file["frequency"] / 50
        data_file["freq_squared_ratio"] = np.power(data_file["frequency"] / 50, 2)
        data_file["freq_cubic_ratio"] = np.power(data_file["frequency"] / 50, 3)
        k = 500
        data_file["skin"] = (k - data_file["intake_pressure"] - data_file["liquid_rate"]) / data_file["liquid_rate"]

        # Calculating derivatives and statistics
        if COLS_TO_CALC is None:
            COLS_TO_CALC = [
                "current",
                "voltage",
                "active_power",
                "frequency",
                "electricity_gage",
                # "motor_load",    # Mistake in initial data_file, so I don't need it here. Should be resolved some day
                "pump_temperature",
            ]

        windows = [60 * 14 * 3]
        # windows = [60 * 14 * 3, 60 * 14 * 1, 60 * 14 * 7]
        for col in COLS_TO_CALC:
            for window in windows:
                data_file[f"{col}_rol_mean_{window}"] = (
                    data_file[col].rolling(min_periods=1, window=window).mean()
                )
                data_file[f"{col}_rol_std_{window}"] = (
                    data_file[col].rolling(min_periods=1, window=window).std()
                )
                data_file[f"{col}_rol_max_{window}"] = (
                    data_file[col].rolling(min_periods=1, window=window).max()
                )
                data_file[f"{col}_rol_min_{window}"] = (
                    data_file[col].rolling(min_periods=1, window=window).min()
                )
                data_file[f"{col}_spk_{window}"] = np.where(
                    (data_file[f"{col}_rol_mean_{window}"] == 0), 0, data_file[col] / data_file[f"{col}_rol_mean_{window}"]
                )
            data_file[f"{col}_deriv"] = pd.Series(np.gradient(data_file[col]), data_file.index)
            # data_file[col] = data_file[col].rolling(min_periods=1, window=30).mean()
            data_file[f"{col}_squared"] = np.power(data_file[col], 2)
            data_file[f"{col}_root"] = np.power(data_file[col], 0.5)

        new_name = filename[:-4] + "_featured.csv"
        save_path = os.path.join(output_path, new_name)
        data_file.to_csv(save_path)
