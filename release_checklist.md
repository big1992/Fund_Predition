# Production Readiness Checklist

## Security
- [ ] `OPENAI_API_KEY` and other secrets are provided via environment variables (not hardcoded).
- [ ] CORS origins are restricted to trusted frontend domains.
- [ ] Dependency scan completed (`pip-audit`, `npm audit`) and critical issues resolved.

## Reliability
- [ ] Backend health endpoint returns `healthy` after startup.
- [ ] Core API routes (`/api/stocks`, `/api/predictions`, `/api/backtest`, `/api/portfolio`) pass smoke tests.
- [ ] Model metadata files are generated after training and can be traced by symbol/model/time.

## Performance
- [ ] Frontend production build succeeds.
- [ ] Backtest and prediction endpoints respond within acceptable latency under expected load.
- [ ] Large model artifacts are persisted on mounted storage (`backend/saved_models`).

## Data Safety
- [ ] SQLite database backup policy defined and tested.
- [ ] Recovery test performed from latest backup.
- [ ] Migration plan defined if moving from SQLite to managed DB.

## Deployment
- [ ] `docker compose up --build` runs both frontend and backend successfully.
- [ ] Frontend reachable at `http://localhost:5173`.
- [ ] Backend docs reachable at `http://localhost:8000/docs`.

## Go/No-Go
- [ ] All Sprint issues for release are moved to `Done`.
- [ ] Known blockers documented and accepted.
- [ ] Release owner approved go-live.
