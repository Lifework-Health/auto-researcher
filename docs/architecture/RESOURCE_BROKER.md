# ResourceBroker boundary

`auto_researcher.resources` owns operational decisions about **where** and
**when** work may run. It does not accept an experiment, candidate
configuration, dataset, split, seed, evaluator, metric, objective, or score.
Those remain under the existing task, search, and evaluator boundaries.

## Contracts

- `ResourceRequirement` describes a typed resource, quantity, and generic
  scalar capacity floors. A GPU request uses resource type `gpu`, quantity for
  the GPU count, and `memory_mib` for minimum available VRAM. The same model can
  later express CPU, RAM, scratch/storage, cluster slots, API/evaluator quota,
  or cloud-spend capacity without adding task concepts.
- `ResourceRequest` adds admission class, priority, maximum wait, stable-idle
  duration, pre-emption semantics, and equivalence requirements. Pre-emption is
  fail-closed and unimplemented; `never` is the default.
- `ResourceProvider` reports current typed `ResourceCandidate` observations.
  Invalid or incomplete provider state rejects admission rather than guessing.
- `ResourceAdmissionPolicy` decides admit/wait/reject from only the request and
  current resource observation. `CourtesyResourceAdmissionPolicy` implements
  capacity, utilization, foreign-owner, and continuous-idle checks.
- `ResourceBroker` polls and re-checks at every acquisition boundary. Busy or
  foreign-owned resources remain an operational wait, not a scientific result.
- `ResourceLeaseStore` is the replacement boundary for later durable shared
  coordination. The first in-memory implementation provides atomic claim,
  worker ownership, acquisition, heartbeat renewal, release, expiry detection,
  and stale recovery.

The current broker admits one resource requirement at a time and fails closed
on a multi-resource bundle. The models retain typed requirement tuples so a
later durable coordinator can add atomic bundles without changing scientific
interfaces. PostgreSQL and coordinated shared workers are intentionally left to
PR 11.5.

## FeTA adapter

`FeTAGPUResourceProvider` maps the existing physical `nvidia-smi` observation
to a generic GPU candidate. `gpu_resource_request` maps the validated FeTA
runtime-only scheduler policy to the generic request. The generic courteous
policy and broker now perform the existing polling loop; the adapter retains:

- strict `CUDA_VISIBLE_DEVICES` binding before admission;
- primary and opportunistic modes;
- minimum free-memory and maximum-utilization thresholds;
- foreign-process courtesy and own-process exclusion;
- continuously-idle stability windows and their reset behavior;
- indefinite waiting for an eligible configured card;
- admission immediately before CUDA seeding/model construction;
- no default pre-emption; and
- legacy FeTA log/error/telemetry fields.

The legacy FeTA telemetry mapping remains for evaluator compatibility. Generic
`ResourceAdmissionTelemetry` is explicitly operational and is returned beside,
not inside, scientific evidence or metrics.

## Scientific isolation invariant

Equivalent resource selection may change admission time, elapsed time,
operational telemetry, cost, and lease identity. It cannot change candidate
identity or configuration, dataset/split, seed, evaluator, metric, objective,
score, or result interpretation. The resource API contains none of those
scientific inputs, and deterministic tests hash the same scientific
configuration across equivalent resource assignments.
