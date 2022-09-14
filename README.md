# esp-failures
Бейзлайн для прогнозирования отказов УЭЦН на основе генерации статистических признаков с помощью tsfresh.

## Описание проекта
- В скрипте **catboost_baseline.py** находится код бейзлайна.
- В папке **configs** находятся конфиги для бейзлайна.
- Для запуска бейзлайна в корне проекта должна быть папка **data** с данными. В **public_test.csv** значения целевой переменной для расчета метрик.
- После обучения файл с весами модели **model.cbm** появится в папке **model**.
- После запуска инференса файл с прогнозом **submission.csv** появится в папке **output**.
- В папке **scripts** лежит скрипт **get_metrics.py** для расчета метрик.
```
.
├── catboost_baseline.py
├── configs
│   ├── column_dtypes.json
│   └── tsfresh_features.json
├── data
│   ├── public_test
│   │   ├── 1d7c8385-04bf-49a9-90d7-84c4763d4fae.csv
│   │   ├── ...
│   │   └── f43b876e-9355-481c-badd-9b790db70550.csv
│   ├── public_test.csv
│   └── train
│       ├── 0a0b1d5e.csv
│       ├── ...
│       └── ffffb726.csv
├── Dockerfile
├── LICENSE
├── model
│   └── model.cbm
├── output
│   └── submission.csv
└── scripts
    └── get_metrics.py
```

## Сборка docker-образа
```bash
docker build --tag catboost-baseline:latest .
```

## Запуск обучения в docker-контейнере
```bash
docker run -it --rm -v `pwd`/data/train:/data -v `pwd`/model:/model -v `pwd`/output:/output catboost-baseline:latest \
    --mode=train \
    --data-dir=/data \
    --model-dir=/model
```

## Запуск инференса в docker-контейнере
```bash
docker run -it --rm -v `pwd`/data/public_test:/data -v `pwd`/model:/model -v `pwd`/output:/output catboost-baseline:latest \
    --mode=predict \
    --data-dir=/data \
    --model-dir=/model \
    --output-dir=/output
```

## Запуск расчета метрики в docker-контейнере
Расчет метрики первого этапа:

```bash
docker run -it --rm -v `pwd`/data:/data -v `pwd`/output:/output -v `pwd`/scripts:/scripts \
    --entrypoint python catboost-baseline:latest \
    /scripts/get_metrics.py \
    -g /data/public_test.csv \
    -p /output/submission.csv \
    --target daysToFailure --metric rmsle
```
Расчет метрики второго этапа:

```bash
docker run -it --rm -v `pwd`/data:/data -v `pwd`/output:/output -v `pwd`/scripts:/scripts \
    --entrypoint python catboost-baseline:latest \
    /scripts/get_metrics.py \
    -g /data/public_test.csv \
    -p /output/submission.csv \
    --target CurrentTTF --metric client_val
```
