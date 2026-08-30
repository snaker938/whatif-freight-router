$env:BACKEND_INTERNAL_URL = "http://localhost:8000"
if (-not (Test-Path "node_modules")) { pnpm install }
pnpm dev
