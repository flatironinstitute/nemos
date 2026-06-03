"""Register a custom option in root."""

def pytest_addoption(parser):
    parser.addoption(
        "--timeit",
        action="store_true",
        default=False,
        help="Show aggregated parametrized test durations",
    )
