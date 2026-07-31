# Knowledge reference model

## Records and authority

A `KnowledgeEntity` is a typed concept with a stable CURIE-like identifier,
safe display properties and source links. It never uses a Neo4j internal ID.

A `KnowledgeAssertion` is a directed subject–predicate–object record with
method, source links, asserted-by identity, trust tier and bounded confidence.
Assertions are provider output, not automatically accepted evidence.

A `KnowledgeSource` identifies an ontology release, curated database,
publication or assertion corpus using version, PMID, DOI, accession or CURIE.
Source identity and version are required according to task policy.

A `KnowledgeReference` is the compact, citable projection of one accepted
assertion. It includes a deterministic reference ID, concise claim, endpoint
CURIEs, source IDs, citation label, relevant task parameters, trust tier,
confidence, bundle ID and prior cap. Agents receive references—not arbitrary
graph rows.

## Trust tiers and prior caps

Platform ceilings are:

| Trust tier | Maximum prior weight |
|---|---:|
| `CURATED` | 0.9 |
| `CORPUS` | 0.7 |
| `LIVE` | 0.3 |
| `UNVERIFIED` | 0.3 |

A task may choose a stricter ceiling. `LIVE`, `UNVERIFIED` and LLM-asserted
records cannot qualify for `KNOWLEDGE_GROUNDED` in PR 5 even when present for
diagnostics. With multiple qualifying citations, reconciliation uses the
minimum cited cap. This is a bounded research prior, not a probability that the
hypothesis is true.

## Grounding versus experimental evidence

A completed bundle alone does not ground a proposal. The proposal must cite a
permitted reference from the active bundle; the reference must survive
validation and be relevant to its predicted or planned parameters. Invented,
cross-bundle and uncited facts do not gain authority from model pretraining.

`KNOWLEDGE_GROUNDED` describes the provenance of a hypothesis or plan.
`SUPPORTED`, `REFUTED` and `INCONCLUSIVE` describe measured experimental
evidence. Knowledge retrieval can never mark an experiment supported; only the
task evaluator and deterministic verifier can determine that status.

## Bundle identity

`KnowledgeBundle` contains the graph snapshot metadata, sources, entities,
accepted assertions, references, validation summary, safe artefact references
and a canonical content hash. `KnowledgeBundleReference` is the small
checkpoint-safe pointer. Context assembly reloads the completed bundle from the
retrieval store and verifies its hash before exposing any reference.
