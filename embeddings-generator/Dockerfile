FROM python:3.8-slim-buster

WORKDIR /app

RUN apt-get update
RUN apt-get clean

RUN pip install --upgrade pip
COPY app/requirements.txt ./
RUN pip install -r requirements.txt

COPY app ./app/
EXPOSE 80
CMD uvicorn app.main:app --workers 1 --host 0.0.0.0 --port 80 --log-level 'info'
