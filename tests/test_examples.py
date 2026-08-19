import runpy
from pathlib import Path

import pytest

EXAMPLE_PATHS = sorted((Path(__file__).parents[1] / "examples").glob("*.py"))


@pytest.mark.parametrize("example_path", EXAMPLE_PATHS, ids=lambda path: path.stem)
def test_example_runs(example_path: Path) -> None:
    runpy.run_path(str(example_path))
