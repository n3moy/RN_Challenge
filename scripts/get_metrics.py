import argparse

import pandas as pd
import numpy as np

from pathlib import Path


def get_mape(y_true, y_pred):
    return (abs(y_true-y_pred)/y_true).mean()


def get_smape(y_true, y_pred):
    sapes = (abs(y_true-y_pred)/((abs(y_true)+abs(y_pred))/2)).sum()
    return 100*sapes/len(y_true)


def get_rmsle(y_true, y_pred):
    return np.sqrt(np.mean((np.log(y_pred+1)-np.log(y_true+1))**2))


_DEFAULT_RANGES = [(0, 30, 5),  # Диапазон от 0 до 30 +- 5 суток
                   (30, 180, 18),
                   (180, 365, 36),
                   (365, 730, 73)
                   ]


def get_client_val(y_true, y_pred):
    rel_score = relative_tp_fp_score(y_true, y_pred)
    return rel_score


def relative_tp_fp_score(
        y_true: np.array,
        y_pred: np.array,
        range_deviation_info: list = _DEFAULT_RANGES,
        consider_max_error_as_tp=True):
    """
    Вычислить TP/TP+FP относительно значения целевой переменной
    y_true - массив истинных значений
    y_pred - массив предсказанных значений
    range_deviation_info - массив из кортежей вида (начало диапазона, конец диапазона, максимальное отклонение для диапазона)
    unknown_range_behavior - если True, то для любого значения из y_true, не входящего ни в один диапазон
    максимальное отклонение будет определяться через параметр unknown_range_deviation
    unknown_range_deviation - максимальное отклонение, чтобы считать прогноз TP для значений y_true не входящих ни в один диапазон
    consider_max_error_as_tp - если True, то отклонение меньше или равное максимальному еще будет считаться TP, иначе 
    отклонение должно быть строго меньше (<) максимально допустимого
    """
    def get_range(true_value):
        # Пытаемся обнаружить диапазон, к которому принадлежит y_true
        # Если такой есть - вернуть его индекс
        # В противном случае вернуть -1
        try:
            return [int(true_value >= target_range[0] and true_value < target_range[1]) for target_range in range_deviation_info].index(1)
        except ValueError:
            return np.nan

    def is_true_positive(yt, yp):
        # Получаем индекс диапазона
        target_range_index = get_range(yt)

        if pd.isna(target_range_index):
            return np.nan

        allowed_deviation = range_deviation_info[target_range_index][-1]

        # В зависимости от того считать максимальное отклонение TP или нет
        # посчитать является ли предсказание TP или FP
        if consider_max_error_as_tp and abs(yt-yp) <= allowed_deviation:
            return True
        if (not consider_max_error_as_tp) and abs(yt-yp) < allowed_deviation:
            return True

        return False

    # Для всех значений считаем являются они TP или FP
    # Проводим расчет метрики, возвращаем значение
    positives = np.array([is_true_positive(yt, yp)
                         for yt, yp in zip(y_true, y_pred)
                         if not pd.isna(yt) and not pd.isna(yp)])
    return len(positives[positives == 1])/len(positives)


def main(args):
    arg_fun_map = {
        "mape": get_mape,
        "smape": get_smape,
        "rmsle": get_rmsle,
        "client": get_client_val,
    }

    metric = arg_fun_map.get(args.metric, None)

    if metric is None:
        raise ValueError(f"Метрика {args.metric} не поддерживается!")

    df_true = pd.read_csv(args.ground_truth)
    df_pred = pd.read_csv(args.predictions)

    if len(df_true) != len(df_pred):
        raise ValueError("Разный размер файлов прогноза и истинных значений!")

    df_pred = df_pred = df_pred.rename(columns={
            'filename': 'randomizedName',
            'daysToFailure': 'daysToFailure_pred'
        })

    size_before = len(df_true)
    df_true = df_true.merge(df_pred, on='randomizedName')

    if len(df_true) != size_before:
        raise ValueError("В файле истинных значений и прогноза не совпадают имена файлов!")

    if args.target == "CurrentMTTF":
        df_true['CurrentMTTF_pred'] = df_true['daysFromLastStart'] + df_true['daysToFailure_pred']
        print(metric(df_true['CurrentMTTF'], df_true['CurrentMTTF_pred']))
    else:
        print(metric(df_true['daysToFailure'], df_true['daysToFailure_pred']))


def get_args(args=None):
    parser = argparse.ArgumentParser(
        description='Скрипт для расчета метрик.'
    )
    parser.add_argument(
        '-g', '--ground-truth',
        type=Path, required=True,
        help='Путь к test.csv'
    )
    parser.add_argument(
        '-p', '--predictions',
        type=Path, required=True,
        help='Путь к submission.csv'
    )
    parser.add_argument(
        '-t', '--target',
        type=str, default='daysToFailure',
        help='ЦП для метрики: daysToFailure или CurrentMTTF'
    )
    parser.add_argument(
        '-m', '--metric',
        type=str, default='smape',
        help='Метрика для расчета. Варианты: mape, smape, rmsle, ' +
              'client_val (метрика заказчика)'
    )
    return parser.parse_args(args)


if __name__ == '__main__':
    main(get_args())
