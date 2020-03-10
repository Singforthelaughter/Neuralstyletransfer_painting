FROM python:3.7

RUN mkdir src
WORKDIR /src
COPY . /src

Run pip install --upgrade pip
RUN pip install -r requirements.txt

EXPOSE 5000

ARG SERVICE_FILE=service.py

ENV FLASK_APP=$SERVICE_FILE
ENV FLASK_DEBUG=0

ENTRYPOINT ["python", "-m", "flask", "run", "--host", "0.0.0.0"]