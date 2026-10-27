# Running the tests

## Setup (once)
```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pip install -e ".[dev]"
```

## Run all tests
```powershell
pytest
```

## Run one file or one test
```powershell
pytest tests\test_import.py
pytest -k fixtures
```

## Test files
Tests use the small files in `tests/fixtures/`, never the full dataset.
In a test, get a fixture file with:

```python
def test_example(fixture_path):
    clip = fixture_path("clip.wav")
```

The fixtures are defined in `tests/conftest.py`.

## Continuous integration
Every pull request runs `pytest` on GitHub Actions
(`.github/workflows/tests.yml`). A pull request should only be merged
when the check is green.