import logging
import random

import pytest


@pytest.fixture(scope='function')
def fix_seed():
    seed = 1234
    random.seed(seed)
    logging.info("Fix random seed: {seed}")
