# OpenEvolve model-call identity

The call ID is SHA-256 over a canonical JSON, domain-separated envelope named
`auto-researcher-openevolve-model-call`. It binds run and thread IDs, contract ID
and hash, task ID/version, search request, generation, parent candidate,
component ID/version/interface hash, adapter ID/version, mutation operator,
provider, exact model, prompt ID/version/hash, structured input hash, response
schema, approval ID/hash, executor evidence, and model-budget identity.

The semantic key binds the lifecycle position independently of the payload. A
second payload at that position fails with `model_call_identity_conflict`.
Neither identity uses Python hashing, wall-clock time, or a random UUID.

Completion identity is a second domain-separated canonical hash over call ID,
input/output hashes, approval hash, provider, and model. Reuse checks these
fields and the structured response hash before returning stored output.
