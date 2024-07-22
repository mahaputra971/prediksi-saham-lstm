jupyter:
	venv/bin/jupyter-lab --no-browser --port 3030

fastui:
	fastapi dev main.py

fastui2: 
	uvicorn app.main:app --reload