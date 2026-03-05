FROM python:3.10-slim-buster 


RUN apt-get update -y && \
    apt-get install -y libgomp1 awscli && \
    apt-get clean


WORKDIR /app

COPY . /app

RUN pip install  -r requirements.txt

EXPOSE 8000

CMD ["python3", "app.py"]