FROM python:3.10-slim
WORKDIR /app
COPY . /app/
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
CMD [ "python3 main.py" ]
