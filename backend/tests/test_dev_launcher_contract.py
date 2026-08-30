from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def _nonblank_lines(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_dev_launchers_are_readable_and_preserve_commands() -> None:
    dev_path = REPO_ROOT / "scripts" / "dev.ps1"
    backend_path = REPO_ROOT / "scripts" / "dev-backend.ps1"
    frontend_path = REPO_ROOT / "scripts" / "dev-frontend.ps1"

    dev_content = dev_path.read_text(encoding="utf-8")
    combined_content = "\n".join(
        [
            dev_content,
            backend_path.read_text(encoding="utf-8"),
            frontend_path.read_text(encoding="utf-8"),
        ]
    ).casefold()

    for encoded_token in ("-encodedcommand", "tobase64string", "frombase64string"):
        assert encoded_token not in combined_content

    assert dev_content.count("Start-Process pwsh") == 2
    assert dev_content.count('"-NoExit"') == 2
    assert dev_content.count('"-File"') == 2
    assert r'"..\scripts\dev-backend.ps1"' in dev_content
    assert r'"..\scripts\dev-frontend.ps1"' in dev_content
    assert 'Test-PortListening -Port 8000' in dev_content
    assert 'Test-PortListening -Port 3000' in dev_content
    assert 'Port 8000 already in use. Skipping backend launch.' in dev_content
    assert 'Port 3000 already in use. Skipping frontend launch.' in dev_content

    assert _nonblank_lines(backend_path) == [
        '$env:OSRM_BASE_URL = "http://localhost:5000"',
        '$env:OUT_DIR = "out"',
        'if (-not (Test-Path ".venv")) { uv sync --dev }',
        'uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000',
    ]
    assert _nonblank_lines(frontend_path) == [
        '$env:BACKEND_INTERNAL_URL = "http://localhost:8000"',
        'if (-not (Test-Path "node_modules")) { pnpm install }',
        'pnpm dev',
    ]
