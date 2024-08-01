FROM python:3.12
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install uvicorn
COPY . .
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--reload"]