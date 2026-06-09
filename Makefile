.PHONY: install demo run test clean amd-setup

install:
	pip install -r requirements.txt
	cp -n .env.example .env || true

demo:
	python scripts/generate_demo.py --company "Apple Inc" --backend mock

run:
	python scripts/run_pipeline.py --company "$(COMPANY)" --backend $(or $(BACKEND),mock)

test:
	pytest tests/ -v

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf data/cache/* data/outputs/*

amd-setup:
	bash scripts/setup_amd.sh

notebook:
	jupyter lab notebooks/
