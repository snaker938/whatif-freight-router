$env:OSRM_BASE_URL = "http://localhost:5000"
$env:OUT_DIR = "out"
if (-not (Test-Path ".venv")) { uv sync --dev }
uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
