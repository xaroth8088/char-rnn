FROM tensorflow/tensorflow:2.2.1-gpu

WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
