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
│   ├── public_test
│   │   ├── 0ba1977c-a858-4cc0-a629-34d75c83b972.csv
│   │   ├── 5dba9d1a-fab2-49c0-ad11-65eee33d714c.csv
│   │   └── fd9072b8-4f64-40f2-b9f5-503913aedf88.csv
│   ├── public_test.csv
│   └── train
│       ├── 900006.csv
│       ├── 900008.csv
│       └── 900011.csv
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
```bash
docker run -it --rm -v `pwd`/data:/data -v `pwd`/output:/output -v `pwd`/scripts:/scripts \
    --entrypoint python catboost-baseline:latest \
    /scripts/get_metrics.py \
    -g /data/public_test.csv \
    -p /output/submission.csv
```
