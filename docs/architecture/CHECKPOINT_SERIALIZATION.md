# Checkpoint serialization trust boundary

Auto Researcher persists LangGraph state with `JsonPlusSerializer` and an
explicit qualified-type allowlist. The allowlist is part of the checkpoint
security boundary: it permits reconstruction of known immutable domain values
without permitting arbitrary module imports, wildcard modules, pickle fallback,
or checkpoint-controlled callable dispatch.

`ResearchState` persists a `ResearchContract`. Since knowledge-read-safety-v2,
that contract contains
`KnowledgeGroundingRequirement.permitted_read_safety_modes`, whose members are
instances of `auto_researcher.contracts.enums.ReadSafetyMode`. The exact enum is
therefore allowlisted. Restoring it as the loose string
`"OPERATOR_ATTESTED"` changes the canonical contract hash and must fail closed.
Canonical hashing recursively preserves typed sets from Pydantic models. Enum
sets use their closed declaration order, while primitive sets use canonical
lexical order, so reconstruction is independent of `PYTHONHASHSEED` and remains
compatible with the checkpoint 04c identity.

The only type added by corrective PR 5.7 is:

```text
auto_researcher.contracts.enums.ReadSafetyMode
```

Operator-attestation objects are runtime provider configuration, not LangGraph
state. Their platform, service-tier, identity-class, credential-class,
prohibited-capability, and residual-risk enums are persisted in the separate
knowledge-retrieval store through validated JSON values. They are deliberately
not checkpoint-allowlisted.

During execution-identity reconstruction, the stored contract and execution
identity must be their exact allowlisted model classes, and every permitted
read-safety mode must have exact `ReadSafetyMode` identity. Untyped mappings,
string substitutions, subclasses, malformed qualified types, invalid enum
values, arbitrary classes, functions, callables, provider clients, and other
unallowlisted objects cannot become executable checkpoint state.

Reconstructed terminal INSPECT coverage serializes an operator-attested,
REQUIRED-grounding, REAL/REFUTED checkpoint, closes the writer, and reads it in
a fresh process. The regression also proves stable contract, initial-input, and
execution hashes; duplicate START and terminal RESUME guards; and zero graph,
external, scientific, artefact, provenance, checkpoint, or reuse writes.
