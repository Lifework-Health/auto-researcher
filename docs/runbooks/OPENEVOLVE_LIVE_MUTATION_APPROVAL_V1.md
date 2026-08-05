# Approval validation runbook

Create the approval outside the repository, with a short expiry and placeholders
resolved to the exact run, contract, synthetic task, component, pinned adapter,
model, prompt, pricing table, hardened-executor policy, and image digest. An
operator must calculate and bind the canonical approval hash; no command in PR 8
grants approval automatically.

Validate safely:

```text
auto-researcher openevolve approval validate --file /protected/approval.yaml
auto-researcher openevolve approval inspect --file /protected/approval.yaml
```

Both commands print only allowlisted identities and limits. Treat schema,
expiry, hash, scope, or executor mismatches as a stop condition. Never add a live
approval to source control.
