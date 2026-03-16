FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONBUFFERED 1

RUN apt-get update && apt-get clean
RUN pip install --upgrade pip

COPY requirements.txt /root/
RUN pip install -r /root/requirements.txt

COPY app.py utils.py /root/

EXPOSE 5000
CMD ["streamlit", "run", "/root/app.py", "--server.port=5000", "--server.address=0.0.0.0", "--server.headless=true"]
