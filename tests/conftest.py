import random

import pytest
import torch


@pytest.fixture(autouse=True)
def _deterministic_seed():
    """Seed all RNGs per test so data-dependent numerics (e.g. LBFGS line
    search in the round-trip tests) cannot flake in CI."""
    torch.manual_seed(0)
    random.seed(0)
    yield
