FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY Task2 ./Task2
COPY Task3 ./Task3
COPY Task4 ./Task4

EXPOSE 8000

CMD ["uvicorn", "Task4.app:app", "--host", "0.0.0.0", "--port", "8000"]
