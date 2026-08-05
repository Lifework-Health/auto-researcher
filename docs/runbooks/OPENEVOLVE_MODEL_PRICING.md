# OpenEvolve model pricing and accounting

Use an exact, non-floating model ID and the existing `ModelPricing` contract.
Provide input, output, optional cache-write/read rates, currency, and immutable
pricing version. Runtime pricing is never guessed.

Before reservation, the per-call maximum must fit both approval and OpenEvolve
budget ceilings. SQLite reserves one call and maximum cost once per semantic
identity. On completion, provider token counts are mandatory and actual cost is
recomputed by the provider boundary. Replay adds no spend. Unknown outcomes
remain conservatively reserved. Currency, model, or pricing-version mismatch
fails closed.
