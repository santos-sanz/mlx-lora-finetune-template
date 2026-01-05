VENV = .venv
PYTHON = $(VENV)/bin/python
PIP = $(VENV)/bin/pip
STREAMLIT = $(VENV)/bin/streamlit

.PHONY: setup train train-current prepare-data run upload clean-logs clean

run:
	$(STREAMLIT) run app.py

setup:
	python3 -m venv $(VENV)
	$(PIP) install -r requirements.txt
	cp .env.example .env
	@echo "Setup complete. Please edit your .env file."

prepare-data:
	$(PYTHON) scripts/prepare_data.py --input data/raw/dataset.json --output data/processed

train:
	$(PYTHON) scripts/train.py --config configs/default.yaml

train-current:
	$(PYTHON) scripts/train.py --config configs/current.yaml

upload:
	$(PYTHON) scripts/upload_to_hf.py --model outputs/checkpoints/final

clean-logs:
	rm -rf outputs/logs/*
	rm -rf outputs/checkpoints/*

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf build/ dist/ *.egg-info/
