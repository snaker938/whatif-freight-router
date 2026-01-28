# Carbon‑Aware Multi‑Objective Freight Router (Starter v0)

This repo is a starter implementation aligned with the MoSCoW requirements and proposed stack:

- **OSRM** (in Docker) for routing + alternatives
- **FastAPI backend** for: metrics, scenario adjustment (stub), Pareto filtering, batch runs + manifest output
- **Next.js frontend** for: Leaflet map + Pareto scatter chart

## Quick start (Docker)

1. Copy env file:

```powershell
Copy-Item .env.example .env
```
