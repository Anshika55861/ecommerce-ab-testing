.PHONY: setup collect market ab test
setup:
	python -m venv venv && ./venv/bin/pip install -r requirements.txt
collect:
	python -m src.collect
market:
	python -m src.market_analysis
ab:
	python -m src.experiment --source simulate
test:
	pytest -q
