FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip \
  && pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cpu \
  && pip install -r requirements.txt

COPY . .

EXPOSE 10000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "10000"]
