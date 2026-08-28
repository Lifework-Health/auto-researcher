# ADR 022: Research State v1 and evidence boundaries

## Decision

Research State is an append-only, task-agnostic programme journal plus a deterministic reconstructed view. It records programme objectives and context, stable external Evidence Card references, internal experimental observations, future diagnostic observations, hypotheses, uncertainties, planner interpretations, planner decisions, experiment intent, work state, and candidate next actions.

The epistemic classes are separate immutable Pydantic records with closed boundary literals. External evidence is represented only by the identity, version, content hash, and opaque store reference of the Research Intelligence `EvidenceCard`; its claim or brief prose is not copied into Research State. Internal measurements carry exact experiment, candidate, result, metric, fidelity, evaluator, and provenance identities. Planner inference and decisions have their own types and cannot validate as observations. Opaque store and artefact references reject filesystem locations, URIs and credential-like values; they never contain patient data or secrets.

SQLite stores every epistemically meaningful record revision and every state revision. Mutable programme concepts such as hypotheses, uncertainties, experiments, work items, and candidate actions advance by explicit consecutive revisions. The durable append boundary compares every later revision to revision one and rejects changes to hypothesis proposition/origin/motivation, uncertainty question, experiment specification/intent, work-item identity/target, and candidate-action epistemic purpose. Evidence, observations, inferences, and decisions are immutable revision-one records. Reconstructing state selects the latest version of mutable concepts while retaining the complete revision journal and all immutable conclusions and decisions.

Deterministic query semantics answer what was learned, what remains uncertain, the evidence lineage of a conclusion or decision, why an experiment was run, which decision it informed, and which recorded experiment or question directly discriminates competing hypotheses. Lineage retains motivating, directly supporting, refuting, and inference-derived evidence as separate roles. Experiment rationale uses only immutable proposal-time intent and motivating origins, never later results.

## Boundaries

| Record | Epistemic meaning | May be treated as an internal measurement? |
| --- | --- | --- |
| Research Intelligence `EvidenceCard` reference | External attributed evidence | No |
| `InternalExperimentalObservation` | Auto Researcher measurement | Yes |
| `DiagnosticObservation` | Typed internal diagnostic finding | No, unless explicitly queried as diagnostic evidence |
| `ResearchHypothesis` | Testable proposition | No |
| `PlannerInference` | Interpretation derived from cited evidence | No |
| `PlannerDecision` | Chosen action with rationale and cited support | No |

## Consequences

- Research Intelligence remains unchanged and retains the `EXTERNAL_RESEARCH_INTELLIGENCE` boundary.
- Research Intelligence briefs may guide future planning, but brief prose is not authoritative state evidence.
- Historical evidence updates cannot rewrite the evidence identities behind an existing inference or decision.
- Planner-inference-origin hypotheses retain validated inference identities; mixed origins enumerate their constituent origin classes.
- Decisions require a typed basis across evidence, inference, hypotheses, uncertainty, experiments, or programme/budget policy, but need no artificial direct evidence or resource implication.
- Diagnostic scope is expressed with task-agnostic data/model scope references; subject references are optional specialised metadata.
- The store contains references rather than full persisted experiment artefacts.
- Planner v2 execution, orchestration, live model calls, and PostgreSQL remain out of scope.
