# Rollback Notes

## Scope
Use this when a new deploy causes regression in API, frontend, or model-serving behavior.

## Fast Rollback (Git + Compose)
1. Identify last known good commit on `main`.
2. Checkout that commit in deployment workspace.
3. Rebuild and restart:
   - `docker compose down`
   - `docker compose up --build -d`
4. Run smoke check:
   - `pwsh ./scripts/smoke-compose.ps1`

## Data/Model Safety
- Database is mounted from `backend/database` and is not deleted by container rebuild.
- Models are mounted from `backend/saved_models` and remain available after rollback.

## Validation After Rollback
- `GET /api/health` returns `healthy`.
- `GET /api/stocks` returns symbol list.
- Frontend opens and renders dashboard/prediction pages.

## Escalation
- If rollback still fails, freeze deploys and capture:
  - failing commit hash,
  - compose logs (`docker compose logs`),
  - smoke script output.
