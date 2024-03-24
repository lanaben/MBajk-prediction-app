FROM python:3.11-slim
WORKDIR /bike_stands
COPY src/serve/bike_stands.py ./src/serve/
COPY models ./models
COPY data ./data
COPY requirements.txt ./
RUN pip install --trusted-host pypi.python.org -r requirements.txt
EXPOSE 5000
ENV NAME Bikes
CMD ["python", "src/serve/bike_stands.py"]
