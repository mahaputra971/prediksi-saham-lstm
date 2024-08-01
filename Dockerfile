FROM python:3.12
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN mkdir -p /app/picture
RUN mkdir -p /app/picture/accuracy
RUN mkdir -p /app/picture/adj_closing_price
RUN mkdir -p /app/picture/close_price_history
RUN mkdir -p /app/picture/ichimoku
RUN mkdir -p /app/picture/prediction
RUN mkdir -p /app/picture/sales_volume
COPY . .
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--reload"]