# Production OpenEvolve model bridge

`openevolve-model-bridge-v1` keeps provider authority inside Auto Researcher.
The pinned upstream adapter supplies a bounded mutation request; it never
receives a provider object, credentials, pricing, retry authority, persistence,
or environment access. The durable bridge validates the immutable approval and
finite search budget, reserves in `agent-call-store-v2`, acquires dispatch
ownership, constructs the provider, persists the response, and only then returns
the structured mutation envelope for deterministic candidate reconciliation.

The existing agent-call store, pricing model, structured provider protocol,
canonical hashing, and generic model-call provenance are reused. The v2 store
adds optional semantic, approval, budget, dispatch, and completion fields; old
hypothesis and planner records remain readable. Their role-specific prompting
and reconciliation are unchanged.

The runtime dependency factory accepts an injected mutation operator and sandbox
runner. This permits the upstream adapter plus durable bridge and the exact
hardened executor to be selected without a LangGraph branch or provider access
inside upstream.

Production assembly uses `build_approved_live_upstream_runtime`, which verifies
the full adapter hash, executor policy hash, image digest, and all three isolation
results before returning the paired adapter and `HardenedDockerExecutor`. A
local runner is not returned by this path.

No live task, model, executor image, patient dataset, Aura access, iCCA, or MRI
workflow is approved by this infrastructure.
