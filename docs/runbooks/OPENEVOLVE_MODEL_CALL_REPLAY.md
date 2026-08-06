# Durable mutation replay and RESUME

On RESUME, reconstruct the runtime and agent-call store before any provider
factory. An identical `COMPLETED` identity verifies approval, context, input,
output, and completion hashes, then returns the structured envelope without
credentials, provider construction, a new reservation, or budget spend.

`RESERVED` may acquire dispatch once. `DISPATCHING` returns
`model_call_already_dispatching`; `OUTCOME_UNKNOWN` returns
`model_call_outcome_unknown`. Neither is redispatched. Terminal INSPECT reads
the stored state only and must not construct dependencies, write records, or
append provenance.
