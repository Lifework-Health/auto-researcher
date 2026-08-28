# ADR 021: Research Intelligence MVP

## Decision

Research Intelligence is a task-agnostic, offline-first subsystem for already-retrieved external research. `SourceRecord` preserves stable identity and immutable bibliographic/content versions. `SourceRetrievalRecord` separately preserves every observation of a source version, so a later unchanged retrieval cannot erase earlier provenance. Deterministic synthesis creates immutable, traceable `EvidenceCard` records. Support/conflict relationships are derived, durable store state and are reconciled against existing cards as new cards arrive. A separate SQLite store makes all records and refreshes durable. `ResearchIntelligenceBrief` is a derived advisory view and is never the source of truth.

The subsystem has an explicit `EXTERNAL_RESEARCH_INTELLIGENCE` boundary. External claims are not evaluation results, measured experiment evidence, hypotheses, planner inference, or decisions. Applicability and ranking make external evidence useful without promoting it to internally measured fact.

| Record | Meaning | May establish measured performance? |
| --- | --- | --- |
| `SourceRecord` / `EvidenceCard` | External, attributed research evidence | No |
| `EvaluationResult` | Internal evaluator output | Yes, within experiment provenance |
| hypothesis or planner inference | A proposition to test | No |
| decision | An explicit downstream choice | Only under its governing evidence and policy |

## Consequences

- Scouts accept only already-retrieved material; this PR adds no live-web path.
- Identity and ranking are deterministic and versioned; duplicate input is idempotent.
- Contradictions remain addressable rather than being averaged away.
- Every brief statement cites one or more evidence-card identities.
- SQLite is the first durable implementation, not a commitment to a final database.

## Refresh and ranking semantics

An evidence snapshot identifies the complete evidence-card set known for one exact programme context. A refresh identifies a scan event, including its completion time and source-retrieval events. Therefore two scans can point to the same unchanged snapshot while remaining distinct durable refresh events. Each refresh records newly inserted source/content versions and evidence cards separately from the complete snapshot; a no-change refresh has empty `new_*` fields.

`quality_score` is a normalised source-level judgement that must document how trust, source type, and authority within the source's stated context were considered. Source type is not a universal hierarchy: official implementation evidence may be most authoritative for implemented behaviour, while a peer-reviewed study may be stronger for a comparative scientific claim. Trust classification imposes a deterministic maximum source-quality score (`HIGH=1.0`, `MODERATE=0.85`, `LOW=0.55`, `UNVERIFIED=0.25`), preventing an unverified source from receiving an arbitrarily dominant score. Claim-specific strength remains explicit in each finding's `EvidenceQuality` and confidence; ranking v2 combines the trust-capped source score and finding quality with applicability and freshness.

Planner v2, Research State, resource brokering, OpenEvolve integration, a final model search space, and live retrieval remain follow-on work.
