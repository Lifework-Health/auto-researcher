# Upstream adapter and executor troubleshooting

- Dependency unavailable: install the reviewed optional extra outside a run.
- Identity/API/hash mismatch: do not bypass it; compare the pinned release, wheel and lock.
- Hardened executor unavailable/runtime mismatch: start the reviewed Docker engine or stop.
- Image mismatch: rebuild, inspect, review and explicitly update the policy; never use a floating tag.
- Isolation unverified: stop. Do not fall back to the local runner for untrusted mutations.
- Candidate reconciliation failure: inspect only safe envelope codes; do not persist raw provider output.
- Candidate execution failure: use sanitized bounded logs and preserve the immutable run identity.
