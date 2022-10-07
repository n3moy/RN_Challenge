from pathlib import Path
import pandas as pd
import numpy as np
import glob
import json
from typing import List
from tsfresh import extract_features
from tsfresh.feature_extraction import MinimalFCParameters
# from sklearn.linear_model import Lasso
# from sklearn.preprocessing import StandardScaler


Data_folder = "./data/processed"


def count_mean_work_days(dfs):
    print('Mean days of work for wells')
    for df in dfs:
        print(df['FailureDate'].value_counts().mean())


def calc_corelation(dfs):
    print('Correlation for wells')
    all_df_stat = []
    for df in dfs:
        stat = df.describe().T[['mean', 'std', 'min', 'max']]
        stat['std/mean'] = stat['std'] / stat['mean']
        corr = df.corr()['daysToFailure']
        full_stat = stat.merge(corr.to_frame(), 
                               left_index = True,
                               right_index = True)
        full_stat['corr_abs'] = full_stat['daysToFailure'].abs()
        full_stat = full_stat.sort_values('corr_abs', ascending = False)
        #print(full_stat.iloc[:30])
        print(full_stat.head(20))
        all_df_stat.append(full_stat)
    return all_df_stat


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
    save_path: Path = None
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

    if isinstance(save_path, Path):
        print(f"Report saved to: {save_path}")
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

    report_folder = Path(__file__).parent.parent.parent / "reports"
    report_path = report_folder / "lasso_coeffs.xlsx"
    get_top_features(joined_data, save_path=report_path)


def start_lasso_analysis_processed(
    data_dir: Path
):
    from scripts.data.process_df import make_processed_df

    # fnames = data_dir.rglob('*.csv')
    cfg_dir = Path(__file__).parent.parent.parent / "configs"
    column_dtypes = read_cfg(cfg_dir / "column_dtypes.json")
    tsfresh_features = read_cfg(cfg_dir / "tsfresh_features.json")

    train_df, _ = make_processed_df(data_dir, 'train', 4, column_dtypes, tsfresh_features)
    cttf = train_df["CurrentTTF"].astype(int)
    target = train_df["daysToFailure"].astype(int)
    # train_df[['CurrentTTF', 'daysToFailure']] = train_df[['CurrentTTF', 'daysToFailure']].astype(int)
    train_df = train_df.select_dtypes(include=np.number)
    train_df = train_df[tsfresh_features].fillna(method='ffill')
    train_df = train_df[tsfresh_features].fillna(method='bfill')
    train_df = train_df[tsfresh_features].fillna(value=-1)
    train_df["CurrentTTF"] = cttf
    train_df["daysToFailure"] = target
    report_folder = Path(__file__).parent.parent.parent / "reports"
    report_path = report_folder / "lasso_coeffs_tsfresh.xlsx"
    get_top_features(train_df, save_path=report_path)


def get_catboost_model_feature_imp(model_dir: Path,
                                   cfg_dir: Path,
                                   well_path: Path,
                                   save_dir: Path):
    from catboost import CatBoostRegressor

    tsfresh_features = read_cfg(cfg_dir / 'tsfresh_features.json')
    model = CatBoostRegressor().load_model(model_dir / 'model.cbm')
    features = get_catboost_features(cfg_dir, well_path)
    out_df = pd.DataFrame(index=features)
    out_df["Importance"] = np.abs(model.get_feature_importance())
    out_df = out_df.sort_values(by="Importance", ascending=False)

    save_name = "catboost_coeffs.xlsx"
    save_path = save_dir / save_name
    print(out_df)
    out_df.to_excel(save_path, index=True)


def get_catboost_features(cfg_dir, well_path):
    # df = pd.read_csv(well_path, low_memory=False, dtype=column_dtypes)
    column_dtypes = read_cfg(cfg_dir / 'column_dtypes.json')
    tsfresh_features = read_cfg(cfg_dir / 'tsfresh_features.json')
    df = pd.read_parquet(well_path)

    df['SK_Calendar'] = pd.to_datetime(df['SK_Calendar'], format='%Y-%m-%d')
    df['lastStartDate'] = pd.to_datetime(df['lastStartDate'], format='%Y-%m-%d')

    df = df.merge(
            df[['SK_Well', 'SK_Calendar']].groupby('SK_Well').min().rename(
                columns={'SK_Calendar': 'CalendarStart'}
            ),
            on='SK_Well', how='left'
    )

    df['SKLayers'] = df['SKLayers'].fillna(value='').str.split(';').map(len)
    df['CalendarDays'] = (df['SK_Calendar'] - df['CalendarStart']).dt.days

    df[tsfresh_features] = df[tsfresh_features].fillna(method='ffill')
    df[tsfresh_features] = df[tsfresh_features].fillna(method='bfill')
    df[tsfresh_features] = df[tsfresh_features].fillna(value=-1)

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
    X = X.drop(columns = ['CurrentTTF', 'FailureDate', 'daysToFailure'])
    return X.columns


def calc_cat_features_stat(
    input_data: pd.DataFrame,
    save_folder: Path = None,
    top_n: int = None
):
    val_counts = pd.DataFrame()
    if top_n is not None:
        cat_features_names = pd.DataFrame(index=range(top_n))
    for col in input_data.columns:
        col_uniq_val = input_data[col].value_counts()
        col_uniq_val_df = pd.DataFrame(data = {
            'Column name': col,
            'Value': col_uniq_val.index,
            'Counts': col_uniq_val.values}
        )
        val_counts = pd.concat([val_counts, col_uniq_val_df], axis = 0)
        if top_n is not None:
            feature_vals = col_uniq_val.index.values
            slice_len = min(top_n, len(feature_vals))
            feature_vals = list(feature_vals[:slice_len]) + [None]*(top_n  - len(feature_vals))
            cat_features_names[col] = feature_vals
    if save_folder is None:
        return val_counts
    else:
        save_name = "categorical_features_stat.csv"
        save_path = save_folder / save_name
        val_counts.to_csv(save_path, index=False)
        if top_n is not None:
            save_name = "categorical_features_unique_vals.csv"
            save_path = save_folder / save_name
            cat_features_names.to_csv(save_path, index=False)


def cat_features_analysis(
    data_dir: Path,
    use_parquet: bool = False
):
    if use_parquet:
        fnames = data_dir.rglob('*.parquet')
    else:
        fnames = data_dir.rglob('*.csv')
    cfg_dir = Path(__file__).parent.parent.parent / "configs"
    column_dtypes = read_cfg(cfg_dir / "column_dtypes.json")
    joined_data = pd.DataFrame()

    for fpath in fnames:
        if use_parquet:
            data_file = pd.read_parquet(fpath)
        else:
            data_file = pd.read_csv(fpath, low_memory=False, dtype=column_dtypes)

        data_file = data_file.drop(columns= ['CurrentTTF',
                                              'FailureDate',
                                              'daysToFailure',
                                              'SKLayers',
                                              'SK_Well',
                                              'lastStartDate',
                                              'SK_Calendar'])
        data_file = data_file.select_dtypes(include='object')

        joined_data = pd.concat([joined_data, data_file], axis=0)

    report_path = Path(__file__).parent.parent.parent / "reports"
    calc_cat_features_stat(joined_data, save_folder=report_path, top_n=5)


def cols_to_groups(cols: list):
    description_path = Path(__file__).parent.parent.parent / "reports" / "Описание признаков.csv"
    features_description = pd.read_csv(description_path, encoding="utf-8")
    features_description = features_description.rename(
        columns={
            "Название параметра": "par_name",
            "Название столбца": "col_name",
            "Тип данных": "data_type"
        }
    )

    features_groups = features_description["Раздел"].unique().tolist()
    report_out = pd.DataFrame(index=features_groups, columns=["cols_count"])
    report_out["cols_count"] = 0
    dict_out = {}
    group_to_cols = {}
    for group in features_groups:
        vals = features_description.query("Раздел == @group")["col_name"].values.tolist()
        # "Название параметра": "par_name",
        # "Название столбца": "col_name",
        # "Тип данных": "data_type"
        group_to_cols[group] = vals
        for col in cols:
            if col in vals:
                report_out.loc[group, "cols_count"] += 1
                if group in dict_out:
                    dict_out[group].append(col)
                else:
                    dict_out[group] = []
                    dict_out[group].append(col)
    return report_out, dict_out, group_to_cols


def enrich_report(
        report_data: pd.DataFrame,
        stats_in: bool = True
):
    out_report = report_data.copy()

    if stats_in:
        out_report["Initial_feature"] = report_data["Feature"].str.split("__").apply(lambda x: x[0])
        report_cols = out_report["Initial_feature"].values.tolist()
    else:
        report_cols = report_data["Feature"].values.tolist()
    _, b, _ = cols_to_groups(report_cols)
    out_report["group"] = out_report["Initial_feature"].apply(lambda x: [k for k in b.keys() if x in b[k]])
    # out_report["col_name"] = out_report["Initial_feature"].apply(lambda x: features_description.query("col_name == @x")["par_name"].values[0])
    #     out_report["col_type"] = out_report["Initial_feature"].apply(lambda x: features_description.query("col_name == @x")["data_type"].values[0])

    return out_report


def intersection_top(
    reports_paths: List[Path],
    top_n: int = 300,
    out_config_path: Path = Path(__file__).parent.parent.parent / "configs"
):
    initial_cols = []
    processed_cols = []
    for fpath in reports_paths:
        report_file = pd.read_excel(fpath)

        if isinstance(top_n, int):
            report_file = report_file.head(top_n)

        i_cols = report_file["Initial_feature"].values.tolist()
        p_cols = report_file["Feature"].values.tolist()

        initial_cols.append(i_cols)
        processed_cols.append(p_cols)

    inter_top_features_i = initial_cols[0]
    inter_top_features_p = processed_cols[0]

    for ix in range(1, len(initial_cols)):
        inter_top_features_i = np.intersect1d(inter_top_features_i, initial_cols[ix])
        inter_top_features_p = np.intersect1d(inter_top_features_p, processed_cols[ix])
    print(f"Top {top_n} processed features number: {len(inter_top_features_p)}")
    print(f"Top {top_n} initial features number: {len(inter_top_features_i)}")

    _, grouped_cols_i, _ = cols_to_groups(inter_top_features_i)
    # print(grouped_cols_i)
    # grouped_cols_p = cols_to_groups(inter_top_features_p)
    save_path_i = out_config_path / "intersection_rf_lasso.json"
    with open(save_path_i, "w") as j:
        json.dump(grouped_cols_i, j, indent=1, ensure_ascii=False)


def start_lasso_analysis_processed(
    data_dir: Path
):
    from scripts.data.process_df import make_processed_df

    # fnames = data_dir.rglob('*.csv')
    cfg_dir = Path(__file__).parent.parent.parent / "configs"
    column_dtypes = read_cfg(cfg_dir / "column_dtypes.json")
    tsfresh_features = read_cfg(cfg_dir / "tsfresh_features.json")
    train_df, _ = make_processed_df(data_dir, 'train', 4, column_dtypes, tsfresh_features)
    report_path = Path(__file__).parent.parent.parent / "reports"
    get_top_features(train_df, save_folder=report_path)


def get_catboost_model_feature_imp(model_dir: Path,
                                   cfg_dir: Path,
                                   well_path: Path,
                                   save_dir: Path):
    from catboost import CatBoostRegressor

    tsfresh_features = read_cfg(cfg_dir / 'tsfresh_features.json')
    model = CatBoostRegressor().load_model(model_dir / 'model.cbm')
    features = get_catboost_features(cfg_dir, well_path)
    out_df = pd.DataFrame(index=features)
    out_df["Importance"] = np.abs(model.get_feature_importance())
    out_df = out_df.sort_values(by="Importance", ascending=False)

    save_name = "catboost_coeffs.xlsx"
    save_path = save_dir / save_name
    print(out_df)
    out_df.to_excel(save_path, index=True)


def get_catboost_features(cfg_dir, well_path):
    # df = pd.read_csv(well_path, low_memory=False, dtype=column_dtypes)
    column_dtypes = read_cfg(cfg_dir / 'column_dtypes.json')
    tsfresh_features = read_cfg(cfg_dir / 'tsfresh_features.json')
    df = pd.read_parquet(well_path)

    df['SK_Calendar'] = pd.to_datetime(df['SK_Calendar'], format='%Y-%m-%d')
    df['lastStartDate'] = pd.to_datetime(df['lastStartDate'], format='%Y-%m-%d')

    df = df.merge(
            df[['SK_Well', 'SK_Calendar']].groupby('SK_Well').min().rename(
                columns={'SK_Calendar': 'CalendarStart'}
            ),
            on='SK_Well', how='left'
    )

    df['SKLayers'] = df['SKLayers'].fillna(value='').str.split(';').map(len)
    df['CalendarDays'] = (df['SK_Calendar'] - df['CalendarStart']).dt.days

    df[tsfresh_features] = df[tsfresh_features].fillna(method='ffill')
    df[tsfresh_features] = df[tsfresh_features].fillna(method='bfill')
    df[tsfresh_features] = df[tsfresh_features].fillna(value=-1)

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
    X = X.drop(columns = ['CurrentTTF', 'FailureDate', 'daysToFailure'])
    return X.columns


# --- MAIN ---


if __name__ == "__main__":
    DO_CORR = False
    DO_LASSO = False
    DO_CATBOOST = False
    DO_CAT_ANALYSIS = True
    DO_FEATURE_INTERSECTION = False

    if DO_CORR:
        files = get_parquet_files(Data_folder)
        dfs = []
        for file in files:
            dfs.append(pd.read_parquet(file))
        count_mean_work_days(dfs)
        calc_corelation(dfs)

    if DO_LASSO:
        DATA_DIR = Path(__file__).parent.parent.parent / "data" / "processed"
        DO_PROCESSING = True
        if DO_PROCESSING:
            start_lasso_analysis_processed(DATA_DIR)
        else:
            start_lasso_analysis(DATA_DIR)

    if DO_CATBOOST:
        MODEL_DIR = Path(__file__).parent / "old_models"
        CFG_DIR = Path(__file__).parent.parent.parent / "configs"
        WELL_PATH = Path(__file__).parent.parent.parent / "data" / "processed" / "000bb919.parquet"
        SAVE_DIR = Path(__file__).parent.parent.parent / "reports"
        get_catboost_model_feature_imp(MODEL_DIR, CFG_DIR, WELL_PATH, SAVE_DIR)

    if DO_CAT_ANALYSIS:
        DATA_DIR = Path(__file__).parent.parent.parent / "data" / "processed"
        cat_features_analysis(DATA_DIR, use_parquet=False)

    if DO_FEATURE_INTERSECTION:
        reports_path_list = [
            Path(__file__).parent.parent.parent / "reports" / "lasso_coeffs_tsfresh.xlsx",
            Path(__file__).parent.parent.parent / "reports" / "rf_v1_coeffs.xlsx",
            Path(__file__).parent.parent.parent / "reports" / "rf_v2_coeffs.xlsx"
        ]
        intersection_top(reports_path_list)

