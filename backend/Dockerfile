FROM python:3.9-slim-buster

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PORT 8080
EXPOSE 8080

CMD ["uvicorn", "server_enhanced:app", "--host", "0.0.0.0", "--port", "${PORT}"]


