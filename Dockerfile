FROM python:3.9.13

WORKDIR /app
COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt
COPY catboost_baseline.py ./catboost_baseline.py
COPY configs ./configs
COPY model ./model

ENTRYPOINT ["python", "catboost_baseline.py"]

CMD ["--mode=train", "--data-dir=/data", "--model-dir=/model"]
