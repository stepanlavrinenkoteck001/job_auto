FROM python:${PYTHON_VERSION}-alpine
ARG PYTHON_VERSION
ENV PYTHON_VERSION=3.10.9
RUN apk add gcc musl-dev linux-headers python3-dev &&\
    pip install pipenv &&\
    addgroup -S -g 1001 app && adduser -S -D -h /app -u 1001 -G app app

WORKDIR /app

COPY . .

USER app

RUN pipenv --python /usr/local/bin/python3 install

CMD [ "pipenv", "run", "gunicorn", "app:app", "-w", "2", "--threads", "2", "-b", "0.0.0.0:3000" ]