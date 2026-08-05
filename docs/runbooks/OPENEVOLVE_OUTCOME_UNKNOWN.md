# OUTCOME_UNKNOWN operations

Stop the run and preserve all stores. Do not RESUME to obtain another provider
call and do not edit the record. Record only the safe call ID, approval hash,
provider/model identity, timestamps, token/cost fields if known, and public error
code. Never copy raw SDK errors, prompts, responses, headers, or credentials.

Determine externally whether the provider processed the request. Because the
current provider protocol has no verified idempotency key, resolution cannot
rewrite history or trigger an automatic retry. A subsequent call requires an
operator-reviewed, newly approved identity and retains the conservative charge
for the unknown outcome.
