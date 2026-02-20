# Sample Manifest and Outputs

Last Updated: 2026-02-19  
Applies To: `POST /batch/pareto` and run artifact endpoints

## Sample Request/Response Files

- Request: `docs/examples/sample_batch_request.json`
- Response: `docs/examples/sample_batch_response.json`
- Manifest: `docs/examples/sample_manifest.json`

## Typical Retrieval Flow

1. Submit `POST /batch/pareto`.
2. Extract returned `run_id`.
3. Fetch `GET /runs/{run_id}/manifest`.
4. Fetch artifacts from `GET /runs/{run_id}/artifacts`.

## Example Command

```powershell
$body = Get-Content docs/examples/sample_batch_request.json -Raw
Invoke-RestMethod -Uri "http://localhost:8000/batch/pareto" -Method Post -ContentType "application/json" -Body $body
```

## Related Docs

- [Documentation Index](README.md)
- [Backend APIs and Tooling](backend-api-tools.md)
- [API Cookbook](api-cookbook.md)
- [Reproducibility Capsule](reproducibility-capsule.md)
