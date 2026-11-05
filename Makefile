.PHONY: install install-extra install-dev train train-fast train-all tune train-sample test lint predict clean

install:
	pip install -r requirements.txt
	pip install -e .

install-extra:
	pip install -r requirements-extra.txt
	pip install -e .

install-dev:
	pip install -r requirements-dev.txt
	pip install -e .

# Core models: Logistic Regression, Linear SVM, Naive Bayes, Random Forest
train:
	python -m news_classification.train

# Skip the slow Random Forest
train-fast:
	python -m news_classification.train --models lr svm nb

# Everything: core + XGBoost + LightGBM + Keras neural network (needs requirements-extra.txt)
train-all:
	python -m news_classification.train --xgboost --lightgbm --nn

tune:
	python -m news_classification.train --models lr svm nb --tune

# Smoke run on the bundled synthetic sample (never report these numbers)
train-sample:
	python -m news_classification.train --data data/sample/sample_train.csv --test-data data/sample/sample_test.csv --no-update-readme

test:
	pytest

lint:
	ruff check .

predict:
	python -m news_classification.predict "Manchester United beat Chelsea 2-1 in a thrilling Premier League match."

clean:
	rm -rf .pytest_cache .ruff_cache build dist *.egg-info
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +
