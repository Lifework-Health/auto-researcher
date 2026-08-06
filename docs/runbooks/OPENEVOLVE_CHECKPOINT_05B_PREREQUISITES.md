# Checkpoint 05B prerequisites

Preflight the seed-inclusive evaluation budget before creating a live approval.
A one-generation run evaluating one evolved candidate requires exactly two
candidate evaluations: the generation-zero seed baseline and the evolved
candidate. A budget of one must fail with
`openevolve_mutation_evaluation_budget_too_small` before credentials or an
executor are constructed.

Use a dedicated, stable hardened-executor workspace parent. It may initially be
absent and will be materialised by the executor, but it must not be a symlink,
file, repository root, artefact root, approval directory, or candidate-selected
path. Confirm operation children are cleaned while the parent remains.

Before checkpoint 05B, require merged corrective PR 8.1 and a separately passed
checkpoint 05A-C for the exact executor-v2 image/runtime/host, a fresh synthetic-only run and stores, a short
immutable approval, exact provider/model/prompt/pricing identities, one-call and
cost ceilings, protected credentials, and no retry.

First prove offline that a completed fake response survives process termination
before candidate evaluation and resumes without provider credentials. Verify one
reservation/completion/candidate/evaluation/verification and unchanged model
budget. PR 8 itself does not approve an executor image or authorise checkpoint
05A/05B.
