.PHONY: setup train prepare-data upload clean-logs

setup:
	python -m venv venv
	./venv/bin/pip install -r requirements.txt
	cp .env.example .env
	@echo "Setup complete. Please edit your .env file."

prepare-data:
	./venv/bin/python scripts/prepare_data.py --input data/raw/dataset.json --output data/processed

train:
	./venv/bin/python scripts/train.py --config configs/default.yaml

upload:
	./venv/bin/python scripts/upload_to_hf.py --model outputs/adapters/final

clean-logs:
	rm -rf outputs/logs/*
	rm -rf outputs/checkpoints/*

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf build/ dist/ *.egg-info/
