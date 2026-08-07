# Live mutation approval contract

`live-mutation-approval-v1` is a credential-free, immutable, expiring contract.
It binds run/contract/task/component, full pinned adapter hash,
provider/model/prompt/operator,
finite token/call/cost limits, pricing and currency, exact executor evidence,
one mutable file, one exact task-owned live dataset class, reviewer-safe identity,
and residual-risk acknowledgement. The closed allowed classes are `synthetic`
and `public_benchmark`; the latter is limited to fixed public non-patient data
whose evaluation remains host-side.

It explicitly denies Aura, patient data, genuine iCCA, MRI, provider access by
upstream, network, subprocess fallback, retries, package installation, multiple
files, and evaluator/verifier mutation. Runtime requests must be a subset.

The approval hash uses `canonical-json-sha256-v1` with domain
`auto-researcher-live-mutation-approval`, excluding only the hash field itself.
Any material edit requires a new approval. Live files belong outside source
control and must contain no email address or credential.

The additive class extension retains `live-mutation-approval-v1`: existing
synthetic payloads, defaults, hashes, and equality semantics are unchanged. A
public-benchmark approval is a distinct identity and cannot authorise synthetic
or any prohibited class.
