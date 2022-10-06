import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.features.feature_engineering import process_cat_feauters_single_df


def read_cfg(cfg_path):
    with open(cfg_path, 'r') as fin:
        cfg = json.load(fin)
    return cfg


def get_top_features(
        input_data: pd.DataFrame,
        top_n: int = None,
        save_path: Path = None
):
    from sklearn.linear_model import Lasso

    X, y = input_data.drop(columns=["daysToFailure"]), input_data["daysToFailure"]
    lasso = Lasso(random_state=42)

    lasso.fit(X, y)

    coeffs = lasso.coef_
    out_df = pd.DataFrame(index=X.columns)
    out_df["Coefficients"] = np.abs(coeffs)
    out_df = out_df.sort_values(by="Coefficients", ascending=False)

    if isinstance(top_n, int):
        out_df = out_df.head(top_n)

    if isinstance(save_path, Path):
        print(f"Report saved to: {save_path}")
        out_df.to_excel(save_path, index=True)
    else:
        return out_df


def start_lasso_analysis(
        data_dir: Path,
        cat_report_path: Path
):
    fnames = sorted(data_dir.rglob('*.csv'))[:2]
    cfg_dir = Path(__file__).parent.parent.parent / "configs"
    column_dtypes = read_cfg(cfg_dir / "column_dtypes.json")
    joined_data = pd.DataFrame()

    for fpath in fnames:
        data_file = pd.read_csv(fpath, low_memory=False, dtype=column_dtypes)
        target = data_file["daysToFailure"]
        data_file = data_file.select_dtypes(include=["object"])
        # data_file = data_file.fillna(method="ffill")
        # data_file = data_file.fillna(method="bfill")
        # data_file = data_file.fillna(-1)
        data_file["daysToFailure"] = target
        joined_data = pd.concat([joined_data, data_file], axis=0)

    ohe_data, _ = process_cat_feauters_single_df(joined_data,
                                                 cols=None,
                                                 mode="one-hot",
                                                 handle_date=True,
                                                 cat_features_names_file=cat_report_path)

    print("daysToFailure" in ohe_data.columns)

    ohe_data = ohe_data.drop(columns=["SK_Calendar", "lastStartDate", "FailureDate", "CalendarStart", "CalendarDays", "SKLayers"])
    report_folder = Path(__file__).parent.parent.parent / "reports"
    report_path = report_folder / "lasso_cat_coeffs.xlsx"
    get_top_features(ohe_data, save_path=report_path)


if __name__ == "__main__":
    BASE_FOLDER = Path(__file__).parent.parent.parent
    DATA_DIR = BASE_FOLDER / "data" / "processed"
    CAT_FEATURES_NAMES_PATH = BASE_FOLDER / 'reports' / 'categorical_features_unique_vals.csv'

    start_lasso_analysis(DATA_DIR, CAT_FEATURES_NAMES_PATH)
