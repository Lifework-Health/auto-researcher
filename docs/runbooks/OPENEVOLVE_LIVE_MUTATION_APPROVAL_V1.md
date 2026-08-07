# Approval validation runbook

Create the approval outside the repository, with a short expiry and placeholders
resolved to the exact run, contract, task, task-owned dataset class, component,
pinned adapter, model, prompt, pricing table, hardened-executor policy, and image digest. An
operator must calculate and bind the canonical approval hash; no command in PR 8
grants approval automatically.

The closed live-mutation classes are `synthetic` and `public_benchmark`. The
latter means a trusted task plugin declares fixed public, non-patient benchmark
data evaluated only by the host; it does not authorise sending raw rows to the
model or candidate. Iris is the first such task. The approval must match the
task-owned class exactly. Unknown tasks do not receive a default, and Aura,
genuine iCCA, MRI, and patient data remain prohibited. Every live run requires
a fresh, exact operator approval.

Validate safely:

```text
auto-researcher openevolve approval validate --file /protected/approval.yaml
auto-researcher openevolve approval inspect --file /protected/approval.yaml
```

Both commands print only allowlisted identities and limits. Treat schema,
expiry, hash, scope, or executor mismatches as a stop condition. Never add a live
approval to source control.
