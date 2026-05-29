# Release Handoff Checklist

## Pre-release

1. Run backend tests:
- `python -m pytest -q`

2. Run frontend build:
- `cd frontend && npm.cmd run build`

3. Run secret scan:
- `python scripts/scan_secrets.py`

4. Confirm Linear sprint/issues are up to date.

## Delivery Artifacts

1. Prediction exports:
- CSV report endpoint
- PDF report endpoint

2. Commercial docs:
- personas/pricing/demo flow
- compliance boundary
- foundation-model benchmark lab

3. MLOps docs/features:
- model registry
- drift monitoring
- watchlist alerts

## Handoff Notes

1. Product positioning:
- Research analytics, not personalized investment advice.

2. Operations:
- Monitor drift status and retrain flag regularly.
- Review alert feed for symbol-level instability.
