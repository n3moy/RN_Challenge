import pandas as pd
import glob

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

if __name__=='__main__':
    files = get_parquet_files(Data_folder)
    dfs = []
    for file in files:
        dfs.append(pd.read_parquet(file))
    count_mean_work_days(dfs)
    calc_corelation(dfs)