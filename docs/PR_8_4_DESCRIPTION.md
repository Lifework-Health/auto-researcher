# PR 8.4: Stabilise semantic provenance across OpenEvolve cycles

Checkpoint 05B-Lite exposed a producer identity collision after the generation-zero
seed had been prepared and evaluated. `record_provenance()` assigned both the
hypothesis and search lifecycle facts the caller's prefix-stable fallback event ID.
The facts had different semantic keys and correct scientific payload hashes, but the
duplicate database event ID caused the semantic append to fail closed.

Generic scientific lifecycle events now derive their event ID from their existing
`provenance-events-v2` semantic key, matching the deterministic identity approach
already used by OpenEvolve-specific events. Re-recording an unchanged hypothesis or
search request therefore returns the existing event. A changed scientific payload
under the same semantic key still raises `conflicting_semantic_provenance_event`;
the store invariant is unchanged.

`Hypothesis.predicted_subspace` is also recursively frozen so downstream candidate
cycles cannot mutate an already-provenanced scientific fact. Candidate- and
generation-specific facts continue to use the separate OpenEvolve event taxonomy.
This PR does not change the graph topology, evaluator, verifier, provider bridge, or
hardened executor image.
