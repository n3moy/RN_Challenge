from pathlib import Path
import pandas as pd
import numpy as np
import glob
import json
# from sklearn.linear_model import Lasso
# from sklearn.preprocessing import StandardScaler


Data_folder = "./data/processed"


def count_mean_work_days(dfs):
    print('Mean days of work for wells')
    for df in dfs:
        print(df['FailureDate'].value_counts().mean())


def calc_corelation(dfs):
    print('Correlation for wells')
    for df in dfs:
        stat = df.describe().T[['mean', 'std', 'min', 'max']]
        corr = df.corr()['daysToFailure']
        full_stat = stat.merge(corr.to_frame(), 
                               left_index = True,
                               right_index = True)
        full_stat['corr_abs'] = full_stat['daysToFailure'].abs()
        full_stat = full_stat.sort_values('corr_abs', ascending = False)
        print(full_stat.iloc[:30])


def get_parquet_files(folder):
    files = glob.glob(folder+'/*.parquet')
    return files


# --- Get top features report using LASSO


def read_cfg(cfg_path):
    with open(cfg_path, 'r') as fin:
        cfg = json.load(fin)
    return cfg


def get_top_features(
    input_data: pd.DataFrame,
    top_n: int = None,
    save_folder: Path = None
):
    from sklearn.linear_model import Lasso
    from sklearn.preprocessing import StandardScaler

    X, y = input_data.drop(columns=["CurrentTTF", "daysToFailure"]), input_data["daysToFailure"]
    lasso = Lasso(random_state=42)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    lasso.fit(X_scaled, y)

    coeffs = lasso.coef_
    out_df = pd.DataFrame(index=X.columns)
    out_df["Coefficients"] = np.abs(coeffs)
    out_df = out_df.sort_values(by="Coefficients", ascending=False)

    if isinstance(top_n, int):
        out_df = out_df.head(top_n)

    if isinstance(save_folder, Path):
        save_name = "lasso_coeffs.xlsx"
        save_path = save_folder / save_name
        out_df.to_excel(save_path, index=True)
    else:
        return out_df


def start_lasso_analysis(
    data_dir: Path
):
    fnames = data_dir.rglob('*.csv')
    cfg_dir = Path(__file__).parent.parent.parent / "configs"
    column_dtypes = read_cfg(cfg_dir / "column_dtypes.json")
    joined_data = pd.DataFrame()

    for fpath in fnames:
        data_file = pd.read_csv(fpath, low_memory=False, dtype=column_dtypes)

        data_file = data_file.select_dtypes(include=np.number)
        data_file = data_file.fillna(method="ffill")
        data_file = data_file.fillna(method="bfill")
        data_file = data_file.fillna(-1)

        joined_data = pd.concat([joined_data, data_file], axis=0)

    report_path = Path(__file__).parent.parent.parent / "reports"
    get_top_features(joined_data, save_folder=report_path)


# --- MAIN ---


if __name__ == "__main__":
    DO_CORR = False
    DO_LASSO = True

    if DO_CORR:
        files = get_parquet_files(Data_folder)
        dfs = []
        for file in files:
            dfs.append(pd.read_parquet(file))
        count_mean_work_days(dfs)
        calc_corelation(dfs)

    if DO_LASSO:
        DATA_DIR = Path(__file__).parent.parent.parent / "data" / "processed"
        start_lasso_analysis(DATA_DIR)
