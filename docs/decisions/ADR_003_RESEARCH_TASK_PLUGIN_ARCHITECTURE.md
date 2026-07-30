# ADR 003: Scientific domains are research task plugins

- Status: accepted
- Date: 2026-07-30

## Context

Auto Researcher must orchestrate unrelated scientific tasks without embedding
domain concepts in its lifecycle graph. iCCA NBS is the first real integration,
while image segmentation is a representative future extension.

## Decision

Each scientific domain implements the runtime-checkable `ResearchTask` protocol.
An instance-scoped, deterministic registry selects a task by ID and version.
Generic runtime assembly validates readiness and the research contract, then
injects task-supplied configuration normalisation, experiment metadata,
evaluator, dataset manifest, artefact policy, and verification policy into the
existing graph.

`TaskRuntimeContext` is runtime-only. The graph state and provenance receive safe
manifests and stable artefact references, not runtime paths or environment
settings.

## Alternatives considered

- Hard-code iCCA into graph nodes: rejected because domain assumptions would
  become control-plane behavior.
- Maintain one graph per domain: rejected because lifecycle and safety behavior
  would drift.
- Branch by task ID: rejected because every new task would require core edits.
- Copy evaluator code: rejected because scientific formulas would acquire
  multiple owners.
- Pass loose dictionaries everywhere: rejected because plugin compatibility and
  persistence boundaries would be unverifiable.

## Consequences

New tasks add plugin code and registration but do not modify graph topology.
Plugin authors must provide explicit immutable descriptors and manifests and
conform to generic evaluator and policy contracts.
