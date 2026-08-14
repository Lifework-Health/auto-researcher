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
  fail-closed and unimplemented; `never` is the default. Admission class may
  affect configured policy semantics such as stable-idle thresholds. Priority
  is coordination intent for a future scheduler comparing multiple requests;
  the v1 process-local broker evaluates one request and does not provide queue
  ordering or fairness.
- `ResourceProvider` reports current typed `ResourceCandidate` observations.
  A provider-level exception, malformed snapshot, or duplicate candidate ID
  makes the complete snapshot untrustworthy and fails closed. Candidate-level
  invalidity or ineligibility skips that candidate without poisoning other
  equivalent candidates in the same valid snapshot.
- `ResourceAdmissionPolicy` decides admit/wait/reject from only the request and
  current resource observation. `CourtesyResourceAdmissionPolicy` implements
  capacity, utilization, foreign-owner, and continuous-idle checks.
- `ResourceBroker` polls and re-checks at every acquisition boundary. Busy or
  foreign-owned resources remain an operational wait, not a scientific result.
  Continuously idle means continuously *observed* eligible: disappearance from
  one snapshot resets the stability window. Maximum wait uses a monotonic exact
  deadline, shortens the final sleep, and permits a final observation at the
  deadline.
- `ResourceLeaseStore` is the replacement boundary for later durable shared
  coordination. The first in-memory implementation provides atomic claim,
  worker ownership, acquisition, heartbeat renewal, release, expiry detection,
  and stale recovery. Acquisition is idempotent for the same active resource,
  request ID, and worker ID, returning the exact existing lease after a caller
  restart or lost response. A different request or worker conflicts.

`ResourceCandidate` is an indivisible allocation unit or bundle in v1.
`quantity` describes matching capacity in that bundle; acquiring a lease
reserves the whole candidate even when the request needs a smaller quantity.
Providers expose independently allocatable CPU slots, GPUs, or bundles as
distinct candidate IDs. Partial suballocation is deferred. The current broker
admits one resource requirement at a time and rejects unsupported
multi-resource requests before provider inspection. The models retain typed
requirement tuples so a later durable coordinator can add atomic bundles
without changing scientific interfaces. PostgreSQL and coordinated shared
workers are intentionally left to PR 11.5.

Provider-reported `foreign_owners` use typed `ResourceOwner` records with an
explicit identity namespace. They are already classified as foreign by the
provider; generic policy never compares those IDs to lease `worker_id` strings.
The FeTA provider preserves its established rule by excluding the current PID
before emitting `process_pid` ownership records.

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
not inside, scientific evidence or metrics. It records both the configured
stable-idle threshold and the actual continuously observed idle duration used
for admission.

## Scientific isolation invariant

Equivalent resource selection may change admission time, elapsed time,
operational telemetry, cost, and lease identity. It cannot change candidate
identity or configuration, dataset/split, seed, evaluator, metric, objective,
score, or result interpretation. The resource API contains none of those
scientific inputs, and deterministic tests hash the same scientific
configuration across equivalent resource assignments.
