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

New mutation calls use `openevolve-mutation-prompt-v2`. Its structured request
derives the mutable file, allowed files, entry point, interface, source-size
limit, import/dependency allowlists, and schemas from the task-owned component
contract. Empty import or dependency lists are rendered explicitly as `NONE`.
This guidance improves candidate compliance but does not replace authoritative
static validation. Completed v1 records retain their original prompt and input
identity for historical replay; new live approvals must bind prompt v2, and
neither approval version authorises the other.

Production assembly uses `build_approved_live_upstream_runtime`, which verifies
the full adapter hash, executor policy hash, image digest, and all three isolation
results before returning the paired adapter and `HardenedDockerExecutor`. A
local runner is not returned by this path.

Infrastructure alone grants no live approval. V1 remains limited to synthetic
and fixed public non-patient benchmarks. The separate metadata-only v2 path can
assemble an MRI-backed task only when the task explicitly opts in and a fresh
approval binds the exact component exposure, prompt, provider, budget and
verified hardened executor. It never approves model access to MRI or patient
data.
