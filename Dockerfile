FROM python:3.10.6-buster

WORKDIR /app

#COPY taxifare
COPY . /app

# COPY dependencies
# Not done yet
COPY requirements.txt requirements.txt
COPY api api
COPY ML_logic ML_logic
COPY model model
COPY raw_data raw_data

# install dependencies
RUN pip install -r requirements.txt
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

#CMD launch API web server
CMD uvicorn api.fastapi:app --host 0.0.0.0 --port $PORT
