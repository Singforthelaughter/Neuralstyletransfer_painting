FROM python:3.6.0

RUN mkdir src
WORKDIR /src
COPY . /src



Run conda install python=3.6
RUN pip install -r requirements.txt

EXPOSE 5000

ARG SERVICE_FILE=service.py

ENV FLASK_APP=$SERVICE_FILE
ENV FLASK_DEBUG=1

ENTRYPOINT ["python", "-m", "flask", "run", "--host", "0.0.0.0"]