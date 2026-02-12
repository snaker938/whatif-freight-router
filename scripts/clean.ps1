Write-Host "Stopping containers..."
docker compose down

Write-Host "Removing OSRM cached data..."
Remove-Item -Recurse -Force -ErrorAction SilentlyContinue "osrm\data\pbf"
Remove-Item -Recurse -Force -ErrorAction SilentlyContinue "osrm\data\osrm"

Write-Host "Recreating OSRM folders..."
New-Item -ItemType Directory -Force -Path "osrm\data\pbf" | Out-Null
New-Item -ItemType Directory -Force -Path "osrm\data\osrm" | Out-Null
New-Item -ItemType File -Force -Path "osrm\data\pbf\.gitkeep" | Out-Null
New-Item -ItemType File -Force -Path "osrm\data\osrm\.gitkeep" | Out-Null

Write-Host "Done. Run:"
Write-Host "  .\scripts\dev.ps1"
Write-Host "or (full Docker stack):"
Write-Host "  docker compose up --build"
