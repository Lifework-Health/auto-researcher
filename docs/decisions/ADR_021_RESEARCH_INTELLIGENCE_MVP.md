# ADR 021: Research Intelligence MVP

## Decision

Research Intelligence is a task-agnostic, offline-first subsystem for already-retrieved external research. `SourceRecord` preserves bibliographic identity and versioned retrieval provenance. Deterministic synthesis creates immutable, traceable `EvidenceCard` records, preserving supporting and contradictory findings. A separate SQLite store makes sources, cards, relationships, and refreshes durable. `ResearchIntelligenceBrief` is a derived advisory view and is never the source of truth.

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

Planner v2, Research State, resource brokering, OpenEvolve integration, a final model search space, and live retrieval remain follow-on work.
