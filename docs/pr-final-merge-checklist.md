# PR and Final Merge Checklist

## 1) Pre-PR Verification

1. Backend tests
- `python -m pytest -q`

2. Frontend build
- `cd frontend && npm.cmd run build`

3. Secret scan
- `python scripts/scan_secrets.py`

4. Working tree clean
- `git status --short` should be empty

## 2) PR Preparation

1. Confirm target branch
- Source: `codex/sprint2-soo68-confidence-calibration`
- Target: `main`

2. PR title (suggested)
- `feat: complete SOO-63 epic (sprints 1-6) with risk-first product hardening`

3. PR description should include
- Scope summary by sprint
- Validation commands + results
- Security note (secret scan)
- Residual risk and follow-ups

## 3) Review Gate

1. Verify changed areas
- Prediction routes/UI
- Model performance/UI
- Drift/alerts/registry APIs
- Export report endpoints
- Docs and handoff artifacts

2. Confirm no accidental secrets
- Re-run `python scripts/scan_secrets.py`

3. Confirm no unintended binary bloat (if policy requires)
- Review any `.db` / large artifacts in PR file list

## 4) Merge Sequence

1. Rebase or merge latest `main` into working branch if required.
2. Re-run verification (tests/build/secret scan).
3. Merge PR to `main`.
4. Tag release (optional):
- `v1.0.0-risk-first`

## 5) Post-Merge

1. Pull `main` locally.
2. Confirm production/demo startup:
- Backend API up
- Frontend UI up
- Critical endpoints respond

3. Share handoff docs
- `docs/epic-soo63-closeout.md`
- `docs/release-handoff-checklist.md`
- `docs/pr-final-merge-checklist.md`
