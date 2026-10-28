from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    """Folder holding the small test files (tests/fixtures/)."""
    assert FIXTURES_DIR.is_dir(), f"Missing fixtures folder: {FIXTURES_DIR}"
    return FIXTURES_DIR


@pytest.fixture
def fixture_path(fixtures_dir):
    """Return the path to a named file inside tests/fixtures/."""
    def _get(name: str) -> Path:
        path = fixtures_dir / name
        assert path.exists(), f"Missing fixture file: {path}"
        return path
    return _get