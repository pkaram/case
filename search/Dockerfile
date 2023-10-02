FROM python:3.9-slim

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONBUFFERED 1

# Install system dependencies
RUN apt-get update && apt-get clean
RUN pip install --upgrade pip


# Install requirements
COPY requirements.txt /root/
RUN pip install -r /root/requirements.txt

COPY app.py /root/app.py
COPY utils.py /root/utils.py
COPY templates/index.html /root/templates/index.html

EXPOSE 5000
ENV FLASK_APP=/root/app.py
CMD flask run --host 0.0.0.0