# PR 8.5: rejected-candidate continuation

OpenEvolve candidate results are candidate-scoped. When a new proposal becomes
current, the graph clears the generic experiment, evaluation, and verification
fields together with candidate validation and preparation state.

A definite static rejection is an ordinary evolutionary outcome. It consumes
the mutation/model-call and failure budgets, but it does not consume evaluator
or verifier budgets and may be followed by another bounded generation. A
rejected candidate is archived without entering candidate execution or
displacing the best verified candidate.

Candidate recording treats explicit rejection or execution failure as
authoritative even if a caller supplies stale prior-candidate results. The
successful path requires the current preparation, experiment, evaluation, and
verification to share one experiment identity; contradictory state fails with
`openevolve_candidate_result_state_conflict`.

Static validation and the production mutation prompt version are unchanged.
The rendered upstream mutation input communicates the immutable interface and
single mutable file, but does not currently include the component import and
dependency allowlists or a complete static-safety summary. Changing
`openevolve-mutation-prompt-v1` is intentionally outside this lifecycle-only
correction.
