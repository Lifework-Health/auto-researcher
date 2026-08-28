# Changelog

All notable changes to Auto Researcher are recorded in this file.

## [2.2.0] - 2026-08-28

### Added

- A bounded Research Director with durable directives, explicit trigger policy,
  evidence context and independent model-call budgets.
- Research Intelligence and Research State records with source, revision and
  evidence-lineage boundaries.
- Native Optuna parity across supported samplers, pruners, multi-objective
  studies, diagnostics, SQLite and PostgreSQL coordination.
- Bounded OpenEvolve execution with registered mutation surfaces, model-call
  replay, lineage, resume, metadata-only approval and hardened-executor options.
- Managed-secret references and a generic ResourceBroker for durable resource
  ownership and GPU admission.
- FeTA BasicUNet, structural U-Net, DynUNet, attention and transformer-family
  search; staged fidelity; diagnostics; probability ensembling; five-fold
  confirmation; and a separately gated one-time holdout evaluator.
- A frozen ARC Virtual Cell 2026 baseline fixture and fail-closed archival and
  campaign preflights.

### Changed

- Campaign portfolio plans are compiled deterministically from registered task
  capabilities and exact resource allocations.
- Checkpoint restoration, duplicate-candidate handling, continuation fidelity,
  finalisation reserves and cross-process serialization were hardened across
  long-running campaigns.
- Repository identity and package metadata now use the canonical
  `Lifework-Health/auto-researcher` location.

### Scientific and operational boundaries

- FeTA development, five-fold confirmation and final-holdout evaluation remain
  explicitly distinct evidence stages.
- Private datasets, subject-level outputs, checkpoints and challenge payloads
  remain outside Git.
- Literature, model proposals and search suggestions cannot bypass task,
  approval, identity, budget or verification gates.

## [2.1.0] - 2026-07-30

- Initial typed LangGraph control plane, task-plugin architecture, Direct and
  Optuna search, durable checkpoints, provenance and bounded live agents.
