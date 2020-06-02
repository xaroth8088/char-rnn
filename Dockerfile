FROM python:3.7.7-buster
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
