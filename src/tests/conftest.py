"""Test-suite options.

Tests marked `slow` run on the real data under data/ (the full harmonized IA
files, moving_p_agg feature trees) and take minutes, or fail where that data
is absent. They are skipped unless pytest is given --runslow:

    python -m pytest src/tests              # fast suite
    python -m pytest src/tests --runslow    # plus the real-data tests
"""
import pytest


def pytest_addoption(parser):
    parser.addoption("--runslow", action="store_true", default=False,
                     help="also run the slow tests on the real data under data/")


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: runs on the real data under data/; needs --runslow")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        return
    skip_slow = pytest.mark.skip(reason="real-data test; run with --runslow")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
