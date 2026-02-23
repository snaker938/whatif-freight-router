# Sample Manifest and Outputs

Last Updated: 2026-02-23  
Applies To: `POST /batch/pareto`, run manifests, signatures, and artifact endpoints

## Sample Files in Repo

- Request sample: `docs/examples/sample_batch_request.json`
- Response sample: `docs/examples/sample_batch_response.json`
- Manifest sample: `docs/examples/sample_manifest.json`

## Runtime Manifest and Artifact Outputs

On successful batch/scenario compare flows, backend writes:

- manifest: `backend/out/manifests/{run_id}.json`
- scenario manifest: `backend/out/scenario_manifests/{run_id}.json`
- artifact folder: `backend/out/artifacts/{run_id}/`
- provenance stream: `backend/out/provenance/{run_id}.jsonl`

Fixed artifact names:

- results.json
- results.csv
- metadata.json
- routes.geojson
- results_summary.csv
- report.pdf

## Retrieval Flow

1. Submit `POST /batch/pareto`.
2. Extract `run_id` from response.
3. Fetch signed metadata:
   - `GET /runs/{run_id}/manifest`
   - `GET /runs/{run_id}/scenario-manifest`
4. Fetch signatures:
   - `GET /runs/{run_id}/signature`
   - `GET /runs/{run_id}/scenario-signature`
5. Enumerate/download outputs:
   - `GET /runs/{run_id}/artifacts`
   - `GET /runs/{run_id}/artifacts/<name>`

## Example Commands

```powershell
$body = Get-Content docs/examples/sample_batch_request.json -Raw
$resp = Invoke-RestMethod -Uri "http://localhost:8000/batch/pareto" -Method Post -ContentType "application/json" -Body $body
$runId = $resp.run_id

Invoke-RestMethod -Uri "http://localhost:8000/runs/$runId/manifest"
Invoke-RestMethod -Uri "http://localhost:8000/runs/$runId/scenario-manifest"
Invoke-RestMethod -Uri "http://localhost:8000/runs/$runId/signature"
Invoke-RestMethod -Uri "http://localhost:8000/runs/$runId/scenario-signature"
Invoke-RestMethod -Uri "http://localhost:8000/runs/$runId/artifacts"
Invoke-WebRequest -Uri "http://localhost:8000/runs/$runId/artifacts/report.pdf" -OutFile ".\\report_$runId.pdf"
```

## Related Docs

- [Documentation Index](DOCS_INDEX.md)
- [Backend APIs and Tooling](backend-api-tools.md)
- [API Cookbook](api-cookbook.md)
- [Reproducibility Capsule](reproducibility-capsule.md)

