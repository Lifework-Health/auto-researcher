# PR 9.1: permit live OpenEvolve on public benchmarks

The production live-mutation contracts previously accepted only the literal
dataset class `synthetic`, so an otherwise valid Iris Checkpoint 06 context
failed Pydantic validation with `Input should be 'synthetic'`.

This corrective PR adds one closed, identity-bearing class:
`public_benchmark`. A runtime may use it only when the trusted task plugin
implements the optional live-dataset-class capability, the approval and call
context select the exact same class, and every existing task/component/model/
prompt/executor identity also matches. Synthetic remains valid. Unknown tasks
fail closed; Aura, genuine iCCA, MRI, and patient data remain prohibited.

`iris_knn@1.0` is the first public-benchmark task. Its classification means the
fixed public, non-patient data are evaluated only by the trusted host. Raw rows,
fold assignments, and row-level results are not provided to the mutation model
or candidate. No task ID is hard-coded into the generic bridge.

The change is additive and retains `live-mutation-approval-v1`: existing
synthetic approval payloads, defaults, equality checks, and hashes are not
reinterpreted. Dataset class already participates in approval and model-call
identity, and cross-class approvals cannot collide or authorise one another.

Offline fake-production tests cover exact public approval, cross-class and
prohibited-class rejection, unknown and iCCA task rejection, hostile override
resistance, raw-data exclusion, real Iris evaluation and verification,
rejected-candidate continuation, replay, and existing synthetic behaviour. An
opt-in smoke also runs the evolved Iris candidate in the retained hardened
image. The prompt, Iris science, static validator, Docker sources, retained
image, and certified executor policy are unchanged. No live approval or
provider call is made by this PR.
