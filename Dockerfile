FROM python:3.11.7-slim
LABEL authors="zoid79"

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir --upgrade pip

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

#RUN uvicorn main:app --reload
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]