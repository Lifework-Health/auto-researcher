# Research Director literature and knowledge boundary

## Purpose

The Research Director may use external research to choose scientific mechanisms,
but external material is not experimental evidence for the current campaign and
never has authority to create an executable configuration.

## V9 operating model

1. A bounded Literature Scout answers a small, explicit question set from
   traceable HTTPS or DOI sources.
2. The Scout output is stored as a hash-bound brief. Shadow briefs have no live
   influence. A reviewed live brief becomes separately typed, untrusted
   `LITERATURE` landscape evidence.
3. The existing provenance-rich `EvidenceCard` is the canonical knowledge-card
   format. Reviewed cards are ranked and projected into one compact,
   deterministic `ResearchDirectorKnowledgeLibrary`.
4. The library and any reviewed literature brief are embedded in
   `research_director_evidence`; the combined manifest hash is part of the frozen
   campaign configuration.
5. The Director may cite these references in a strategic directive. The Planner
   compiles that intent into registered task fields and exact allocations. The
   Supervisor remains the final deterministic gate.

## V9 launch gates

- Evidence cutoff, source identities, card identities, content version and
  library hash are fixed before action preflight.
- Every card retains conditions, limitations, applicability and source-version
  references.
- Retrieved text is treated as data, never as instructions.
- Duplicate or tampered cards, unknown references, over-budget retrievals and
  manifest mismatches fail closed.
- No changing web results are admitted during a V9 campaign. A new retrieval
  creates a new library version and therefore a new campaign identity.

Live, mid-campaign literature search remains a later capability. It requires a
provenance store for retrieved content, deterministic cache semantics, explicit
refresh triggers and a policy for whether a new evidence snapshot may alter an
active campaign.
