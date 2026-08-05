# ADR 016: OpenEvolve identity and lineage

Status: accepted.

All identities use `canonical-json-sha256-v1` with domain separators. Newline-normalized UTF-8 source has its own hash. A candidate ID binds search request ID, immutable component-interface hash, and source hash, so identical source in the same immutable context has one identity. Mutation reservation identity binds generation, parent, birth index, operator, and random state before an external mutation call.

Lineage records bind candidate, parents, generation, birth index, mutation metadata, source/interface/sandbox identities, validation/preparation/evaluation/verification outcomes, feasibility, and safe artefact references. The population preserves ordered active and archive identities, ordered lineage, deterministic tie-breaking, diversity metadata, budget, random state, and stop state. Duplicate source is recorded without reevaluation and population updates are idempotent.

Candidate search artefacts are published as a staged directory and atomically renamed. The manifest binds every payload hash and the aggregate bundle identity; reuse requires the immutable identities to match. Missing, partial, altered, or incompatible bundles fail closed.
