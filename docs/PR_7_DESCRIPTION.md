# PR 7: Pinned upstream OpenEvolve adapter and hardened no-network executor

Base: `434aa27b8fb46e8b6149ca908160b8ccb44dc805`.

This infrastructure PR adds an optional narrow adapter for upstream OpenEvolve v0.3.2 and a digest-bound Docker candidate executor. Auto Researcher remains authoritative; no live model, Aura, patient-data, genuine iCCA or MRI work was performed.

## Validation report

1. Base commit: `434aa27b8fb46e8b6149ca908160b8ccb44dc805`.
2. Branch: `codex/pr-7-pinned-openevolve-adapter-hardened-executor`.
3. Upstream repository: `https://github.com/algorithmicsuperintelligence/openevolve`.
4. Release/tag: `v0.3.2`.
5. Commit: `411fb59c886c18704caaffb611e17cf9e7d824d2`.
6. Package: `openevolve==0.3.2`; wheel SHA-256 `df998b0731d9c1a80883b4aae452cc43405a3e9c61b46d676d06235b4db49366`; sdist SHA-256 `cd41800ab54734d02a895892615a7f4b9240a6f307c82fc1df7335e89b546599`.
7. Licence: Apache-2.0, compatible with narrow unmodified optional interoperability; attribution added.
8. Dependency strategy: optional `auto-researcher[openevolve]`; no runtime installation.
9. Dependency lock hash: `07a76993f0f3347600d9a753e5da4589ffa8b5c19381b73f74a68baeda1587fd`.
10. Adapter: `upstream-openevolve-adapter-v1`, identity/API/hash validated.
11. Used capabilities: upstream `Program` representation and bounded in-memory population recommendation.
12. Disabled: upstream controller, evaluator, providers, embeddings, network, subprocesses, installation, persistence, resume, budget, stopping, telemetry prompts and scientific judgement.
13. Model bridge: identity-bound exactly-once structured fake/deterministic reservations with provider/model/prompt/output metadata.
14. Upstream never receives a provider client, credentials or provider configuration; direct provider classes are never constructed.
15. Reconciliation: one UTF-8 complete replacement for the declared file; canonical newlines and strict size/path/dependency/provider checks.
16. Candidate mapping: Auto Researcher candidate ID is authoritative; upstream ID is metadata only.
17. Population mapping: upstream recommendation is recorded; PR 6 constrained deterministic ranking and replacement remain final.
18. Persistence: `upstream-openevolve-state-v1` contains validated bounded primitives only.
19. Resume: adapter cursor/counters/IDs reconstruct via the existing checkpoint serializer; no upstream object is checkpointed.
20. Executor: Docker/OCI `openevolve-hardened-executor-v1`.
21. Supported: Docker Desktop 29.3.1 Linux/arm64 fully gated locally; other Docker platforms integration-gated; other runtimes unavailable.
22. Base image: Python 3.12.11 slim-bookworm digest `sha256:519591d6871b7bc437060736b9f7456b8731f1499a57e22e6c285135ae657bf7`; local image `sha256:a7fc72f30d80a5d736f022121c7e3ba512454e38492b289666c92cd1e1b3d7a9`.
23. Network proof: DNS, outbound TCP, loopback and metadata-IP probes all denied inside the real container.
24. Filesystem proof: host sentinel, home/SSH and Docker socket hidden; only read-only input and writable output mounted.
25. Environment: no host credentials inherited; only locale/time settings supplied.
26. Resources: read-only root, non-root UID, dropped capabilities, no-new-privileges, PID/memory/CPU/time/tmpfs/output/log limits and cleanup.
27. Residual risks: Docker/kernel/daemon vulnerabilities, platform variance and architecture-specific image digests.
28. Synthetic demo: pinned adapter proposed canonical tree source, hardened execution succeeded, unchanged evaluator scored 0.84 and verifier passed.
29. Fake cell-biology demo: synthetic non-patient inputs used hardened preparation with no Aura or live model.
30. Internal/adapter equivalence: canonical source, task `ExperimentSpec`, evaluator/verifier semantics and core authority preserved; upstream-only metadata intentionally differs.
31. Resume equivalence: existing PR 6 uninterrupted/resumed lifecycle remains green; adapter state round-trip and exactly-once response reuse pass.
32. Dependency absent: full default suite passed with upstream uninstalled (`356 passed, 4 skipped`).
33. Drift: wrong commit/version/RECORD/lock/API fail closed with stable codes.
34. Checkpoint: exact adapter-state allowlist round-trip passes; arbitrary upstream types are excluded.
35. INSPECT remains the unchanged run-execution-v2 read-only path.
36. Duplicate START and terminal RESUME guards remain unchanged and pass in the regression suite.
37. Hostile adapter envelopes cover multi-file, traversal, dependency and provider/evaluator escape attempts.
38. Real executor tests cover isolation, digest mismatch, candidate preparation and policy binding; PR 6 hostile/resource suite remains green.
39. Full installed suite: `372 passed, 2 skipped` in 12.21 seconds.
40. Live Anthropic and genuine iCCA tests remain explicitly skipped; hardened isolation is a real opt-in gate; no unrelated deselection.
41. Changed-file Ruff check: pass.
42. Changed-file Ruff format check: pass.
43. Repository-wide pre-existing findings are reported separately and not reformatted.
44. Confirmed: no Aura, Claude/Anthropic, patient data, MRI, genuine iCCA or live mutation.
45. Known limitation: bridge reservations are an offline fixture implementation; production live use still requires durable model-call-store wiring and a separately approved image digest.
46. Next checkpoint: repeat image build/isolation and replay checks in a dedicated Linux CI job without live model calls.
47. Subsequent scientific PR: validate one bounded non-patient task under an approved live-mutation protocol before considering iCCA or MRI.
