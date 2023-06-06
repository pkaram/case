FROM python:3.8-slim-buster

RUN apt-get update
RUN apt-get clean

WORKDIR /app
COPY app/requirements.txt ./
COPY app/data/archive/mpst_full_data.csv ./
RUN pip install -r requirements.txt

COPY app/vectorize_items.py ./
CMD ["python", "vectorize_items.py"]
