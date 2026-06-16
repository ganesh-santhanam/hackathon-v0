PYTHON ?= .venv/bin/python
PYTEST ?= .venv/bin/pytest
STREAMLIT ?= .venv/bin/streamlit

.PHONY: install test lint format run-ui eval docker-up docker-config health

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

test:
	PYTHONPATH=src $(PYTEST) -q

lint:
	PYTHONPATH=src $(PYTHON) -m ruff check .

format:
	PYTHONPATH=src $(PYTHON) -m ruff format .

run-ui:
	PYTHONPATH=src $(STREAMLIT) run src/industrial_ai/demo/streamlit_app.py

eval:
	PYTHONPATH=src $(PYTHON) -m industrial_ai.evaluation.harness

docker-config:
	docker compose config

docker-up:
	docker compose up --build

health:
	PYTHONPATH=src $(PYTHON) -m industrial_ai.evaluation.harness
