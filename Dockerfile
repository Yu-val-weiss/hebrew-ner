FROM python:3.8-slim

WORKDIR /app

COPY requirements.txt .

# Install necessary tools and libraries for C++11 support
RUN apt-get update && \
    apt-get install -y build-essential && \
    apt-get install -y gcc g++ && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -r requirements.txt

COPY api_configs/ api_configs/
# to avoid copying over the large models, will mount in docker compose
RUN mkdir /app/fasttext
COPY model/ model/
# to avoid copying over the large models, will mount in docker compose
RUN mkdir /app/ncrf_hpc_configs
COPY utils/ utils/
COPY ncrf_main.py ncrf_main.py
COPY ner_app.py ner_app.py

EXPOSE 5000

ENTRYPOINT ["python", "ner_app.py", "--host", "0.0.0.0", "--port", "5000"]
