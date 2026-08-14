# ADR 024 — PostgreSQL coordination for Optuna and ResourceBroker

Status: Accepted for v2.2 PR 11.5

## Context and implementation audit

The integrated base is `f98c4e64ea0f9e4ab15e8ca3cd0a14abaafe5860`.
Before this change, `search/optuna/storage.py` exposed native
`InMemoryStorage` and SQLite-backed native `RDBStorage` only. The Optuna
backend cached one sampler-bearing Study per runtime to preserve sequential TPE
RNG advancement, tagged each trial with a sequential `slot_index`, recovered the
single RUNNING trial after restart, and rejected more than one RUNNING trial.
It also reconstructed a mutable `Trial` from private `_trial_id` to attach later
metadata. Optuna is pinned to 4.9.0; its public ask/tell signatures accept native
fixed distributions and accept a public `Trial` or trial number at tell time.

The integrated ResourceBroker already separates `ResourceProvider`,
`ResourceAdmissionPolicy`, and `ResourceLeaseStore`. Its in-memory store enforces
one active lease per logical request and per resource and allocates one whole
candidate. The FeTA adapter exposed only the configured `physical_gpu_index`,
used `gpu:N` as both placement and logical request identity, inspected with
`nvidia-smi`, and required the process to have `CUDA_VISIBLE_DEVICES=N`.

## Decision

Optuna remains authoritative for Study, Trial, TrialState, parameter values,
objective values, sampling, `Study.ask()`, and `Study.tell()`. Auto Researcher
does not implement TPE, a proposal queue, liar objectives, or another trial state
machine. The shared worker seam remains:

1. a short PostgreSQL admission transaction;
2. native `Study.ask()`;
3. durable operational worker claim;
4. evaluator and verifier outside Optuna;
5. native `Study.tell()` under a fencing check.

`InMemoryStorage` remains for tests and local use. SQLite native `RDBStorage`
remains deterministic and sequential; configuration rejects SQLite when
`shared_workers` is true. PostgreSQL native `RDBStorage` is the only supported
shared-worker backend. The PostgreSQL driver is optional, so ordinary imports
and offline startup do not require it. A typed operational configuration accepts
host, port, database, username, and a Managed Secrets `SecretReference`. The
resolved password exists only while SQLAlchemy constructs the runtime URL. Safe
references contain only `postgresql:<alias>` and never a password, username,
host, DSN, or secret-provider identifier. Storage credentials do not enter task
configuration, study identity, scientific identity, user attributes,
provenance, Research State, or Research Intelligence.

## Atomic ASK budget and worker fencing

The coordination schema is intentionally small and separate from Optuna's
private tables. A stable study-scoped PostgreSQL advisory transaction lock
serializes only trial-budget validation, native ask, and claim insertion.
Evaluation is never serialized. The study budget row is immutable after first
registration. Every claim is keyed by study and Optuna trial number and records
worker ID, UUID fencing token, database timestamps, expiry, release, and an
immutable report digest. It does not store parameters, objective values, or an
independent TrialState.

All expiry and takeover decisions use PostgreSQL `CURRENT_TIMESTAMP`. An old
worker whose UUID fencing token has been replaced cannot evaluate, tell, fail,
or release on behalf of that trial. Tell and stale reconciliation lock the claim
row, re-check the live token and database-time expiry, call public
`Study.tell(trial_number, ...)`, then release the claim. Infrastructure failure
uses native FAIL and never fabricates a penalty objective. Policy-controlled
replacement, if any, is a new native Optuna trial.

The ASK crash windows are explicit. A crash before ask creates nothing. A crash
after native ask but before durable claim can leave an unclaimed RUNNING trial;
ownership is never guessed. Operators/recovery code must apply a bounded orphan
grace before public-API FAIL reconciliation. A valid `asked_at` user attribute is
preferred because `ask_and_claim_trial` obtains it from PostgreSQL. If the crash
happened before that attribute was durably written, recovery falls back to
Optuna 4.9's naive asking-host-local `datetime_start`. The fallback interprets it
using the recovery host's local-zone rules, so a shared deployment must configure
one worker timezone (UTC is preferred), bound clock skew, and set the orphan grace
above that bound. Durable claim expiry itself never relies on host time. A claimed
crash is recovered only after database-time expiry. Expired resource leases are
independently released by ResourceLeaseStore. Evaluator output reuse remains the
provenance/evaluation store's responsibility; the coordination report digest lets
replay distinguish the exact already-recorded report. If tell committed before
acknowledgement, the terminal native trial plus exact digest makes replay
idempotent.

Optuna 4.9 automatic RDB heartbeat handling is designed around
`Study.optimize()`. It does not automatically heartbeat an external ask/tell
lifecycle, so PR 11.5 does not enable `RDBStorage.heartbeat_interval` or claim
that Optuna recovers these workers. Auto Researcher claims heartbeat explicitly;
stale RUNNING trials are reconciled through public `Study.tell(..., state=FAIL)`.

`CoordinatedOptunaWorker` starts a synchronous claim heartbeat immediately after
ASK admission, then renews the claim in a background loop throughout experiment
construction, resource admission, evaluation, and verification. The default
interval is one quarter of `claim_ttl`; an explicit interval must be positive and
no greater than one third of the TTL. Each heartbeat is a short independent
PostgreSQL transaction, never a transaction held across scientific work. Once a
resource lease is acquired, a second loop renews that exact lease with the same
default and validation rule relative to `resource_lease_ttl`.

The first heartbeat failure is retained by the execution guard. A detected lost
claim or lease prevents subsequent verification and prevents native tell or FAIL
under the superseded token. Specific coordination and lease-ownership errors are
surfaced rather than collapsed into an evaluator error. Both heartbeat loops are
stopped and joined before terminal tell/fail and resource-release teardown, so a
late heartbeat cannot race claim release. Python cannot forcibly stop an
arbitrary injected evaluator already executing when a database or network failure
is detected; the minimum enforced boundary is that it cannot verify or write a
terminal Optuna result after it returns and the guard observes the failure.

## Distributed TPE

Sequential storage retains the exact configured seed and previous sampler-cache
behaviour. Shared workers use native `TPESampler(constant_liar=True)`. Each
runtime's native sampler seed is a hash of the configured study seed, durable
logical `worker_id`, and unique operational `worker_session_id`. The session ID
distinguishes process incarnations so restarting the same logical worker does not
replay its initial in-memory RNG stream. It never enters `SearchRequest`,
`ExperimentSpec`, study attributes, or scientific identity, and tests may inject
it deterministically. Optuna remains the only component suggesting values.
Distributed TPE is order- and schedule-dependent; PR 11.5 neither claims full
duplicate prevention nor schedule-independent reproducibility.

## PostgreSQL resource leases

`PostgresResourceLeaseStore` is the sole shared resource-allocation authority.
Partial unique indexes enforce at most one unreleased lease for a logical
`request_id` and for a physical `resource_id`. Transactions recover database-time
expired rows before acquisition. Exact same-request/same-worker/same-resource
recovery returns the original lease without renewal. If two exact first-time
acquisitions race at INSERT, the losing transaction rolls back and a fresh
transaction reads the authoritative active row; the exact match returns that same
lease ID and timestamps without renewal. A different worker, request, or resource
remains a conflict, and the partial unique indexes remain the final authority.
Release or stale recovery allows ordinary selection again. Allocation remains
whole-candidate with no partial allocation, substitution of an active request,
pre-emption, or second allocation state.

## Equivalent GPUs and scientific identity

The generic NVIDIA provider calls only `nvidia-smi`, enumerates every eligible
physical device as `gpu:0` through `gpu:N`, and creates no CUDA context. Each
candidate is one indivisible GPU with available memory, utilization, classified
compute-process owners, and equivalence tags. Inspection and parsing fail closed.

A concurrent work request has caller-supplied stable logical identity derived
from run/study/trial or another work item—not from `gpu:N`. The broker chooses
the physical resource. A child-process environment is copied and assigned
`CUDA_VISIBLE_DEVICES=N`; process-global environment is never mutated between
workers. Equivalent placement is operational and does not modify
`ExperimentSpec` or scientific identity. Existing FeTA YAML with
`physical_gpu_index: N` retains exact-card semantics. The additive
`gpu_selection: equivalent_pool` path requires a caller-supplied logical request
ID and the coordinated ResourceBroker worker seam.

## Consequences and deferred scope

The standard `run start` and sequential LangGraph Optuna path remains a
single-worker SQLite runtime. PR 11.5 establishes reusable PostgreSQL
storage/claim/resource boundaries and `CoordinatedOptunaWorker.run_one()` for a
separately wired shared-worker runtime; it does not make the ordinary CLI launch
multiple HPO workers. The LangGraph programme remains single-owner per checkpoint
thread, and this PR deliberately avoids another graph or a multiprocess
control-plane rewrite. PostgreSQL must be provisioned and migrated operationally;
application code creates only its small coordination tables and never provisions
a server or edits Optuna tables.

The real PostgreSQL gate exercises the worker seam with a one-second claim TTL
and a 2.3-second evaluator, delayed resource admission beyond the claim TTL,
resource renewal beyond the original lease TTL, a superseded-token loss path,
and both conflicting and exact-idempotent two-process lease races. These tests
demonstrate advancing database heartbeat timestamps, blocked stale takeover,
native COMPLETE plus released claim on success, no verification/tell after known
ownership loss, and one unchanged active lease row for an exact concurrent
acquisition.

PR 11.7 remains responsible for sampler parity, pruning, multi-objective/Pareto,
constraints, and plugin samplers/pruners. Also deferred are multi-GPU trials,
fractional allocation, pre-emption, cloud provisioning, Kubernetes, distributed
LangGraph, OpenEvolve parallelism, and automatic scientific retry policy.
