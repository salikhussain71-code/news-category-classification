import pytest

from news_classification import config


@pytest.fixture(scope="session")
def sample_train():
    assert config.SAMPLE_TRAIN_PATH.exists(), "Run scripts/generate_sample_data.py"
    return config.SAMPLE_TRAIN_PATH


@pytest.fixture(scope="session")
def sample_test():
    return config.SAMPLE_TEST_PATH
