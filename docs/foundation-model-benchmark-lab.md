# Foundation Model Benchmark Lab (Chronos / TimesFM / Lag-Llama)

## Objective

Evaluate foundation time-series models as experimental candidates against existing production baselines.

## Benchmark Protocol

1. Dataset
- Use a representative subset of tracked symbols (e.g., 3-5 symbols).
- Keep train/test split aligned with current walk-forward policy.

2. Baselines
- Naive last price
- Mean return baseline
- Current production ensemble

3. Candidate Models
- Chronos (zero-shot)
- TimesFM (zero-shot)
- Lag-Llama (if runtime allows)

4. Metrics
- MAPE
- Directional accuracy
- Inference latency
- Operational complexity (deps/hardware)

## Decision Rule

1. Promote: consistently better MAPE + directional accuracy with acceptable latency/ops burden.
2. Park: mixed quality or high ops burden.
3. Reject: no clear edge over current ensemble or unstable behavior.

## Current Recommendation

Start as "Park": run as offline experiment first, do not replace production ensemble until evidence is stable.
