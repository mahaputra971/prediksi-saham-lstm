jupyter:
	venv/bin/jupyter-lab --no-browser --port 3030

fastui:
	fastapi dev main.py

fastui2: 
	uvicorn app.main:app --port 8001 --reload	

build: 
	docker build . -t mahaputra971/prediksi-saham-lstm:latest

up: 
	docker compose up -d 

down: 
	docker compose down


