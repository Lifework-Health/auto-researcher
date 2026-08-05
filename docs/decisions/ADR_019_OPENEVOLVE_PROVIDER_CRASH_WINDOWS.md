# ADR 019: OpenEvolve provider crash windows

Status: accepted.

The approved provider abstraction exposes no idempotency-key argument, and the
[official Messages create request](https://platform.claude.com/docs/en/api/messages/create)
documents no idempotency-key body or header parameter. Therefore PR 8 does not
claim exactly-once provider execution.

The store provides exactly-once durable completion reuse and at-most-once
automatic dispatch. `RESERVED` is recoverable because no call began.
`DISPATCHING` is never automatically redispatched. A timeout, uncertain SDK
failure, process failure during invocation, or loss after response but before
completion becomes `OUTCOME_UNKNOWN`; its reserved call and cost remain counted
conservatively. Operator review and a newly approved identity are required.

A provider-factory failure is `FAILED_BEFORE_DISPATCH`, because construction
occurs only after dispatch ownership and invocation is known not to have begun.
A definitive provider rejection is `FAILED_CONFIRMED`. Raw exceptions are not
persisted.
