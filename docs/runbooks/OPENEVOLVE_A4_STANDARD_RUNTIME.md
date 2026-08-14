# Native OpenEvolve A4 through the standard runtime

The A4 SegResNet acceptance run uses the ordinary Auto Researcher lifecycle. It
does not have a Python driver. Start from
`examples/tasks/feta_seg_evolve/openevolve-a4-full-strength-template.yaml`, copy
it into protected run configuration, and replace every absolute placeholder
with the reviewed contract, model-bridge, executor, data, output, workspace, and
state paths for the approved run.

`search.type: OPENEVOLVE` plus
`search.openevolve.native_controller: true` is the typed selector for
`StandardNativeOpenEvolveRuntime` and `EmbeddedOpenEvolveSearch`. `false` or an
omitted field retains the legacy thin controller. Other values are rejected.

## Start

```bash
auto-researcher run start \
  --task feta_seg_evolve \
  --contract /protected/a4/contract.yaml \
  --task-config /protected/a4/openevolve-a4.yaml \
  --run-id feta-segresnet-a4-001 \
  --thread-id feta-segresnet-a4-001-thread \
  --checkpoint-db /protected/a4/state/checkpoints.sqlite \
  --provenance-db /protected/a4/state/provenance.sqlite \
  --agent-calls-db /protected/a4/state/agent-calls.sqlite \
  --knowledge-retrievals-db /protected/a4/state/knowledge.sqlite
```

The runtime assembles the ResearchContract and task, metadata-only approved
durable model bridge, task-owned candidate normalizer, evaluation-reuse-v2
scientific coordinator, evaluator/verifier/evidence boundary, ResourceBroker,
approved model ensemble, and embedded native controller. OpenEvolve output and
checkpoints live under the configured output directory at
`runs/<run-id>/openevolve-native/search-<identity>/`.

## Resume

Use the identical task configuration, run/thread identity, contract, and store
paths. The graph checkpoint resumes the standard CLI lifecycle; the native node
discovers the highest valid OpenEvolve checkpoint and checks the immutable
search envelope before loading population, archive, islands, feature maps,
lineage, and remaining budgets.

```bash
auto-researcher run resume \
  --task-config /protected/a4/openevolve-a4.yaml \
  --thread-id feta-segresnet-a4-001-thread \
  --checkpoint-db /protected/a4/state/checkpoints.sqlite \
  --provenance-db /protected/a4/state/provenance.sqlite \
  --agent-calls-db /protected/a4/state/agent-calls.sqlite \
  --knowledge-retrievals-db /protected/a4/state/knowledge.sqlite
```

Stop before a paid call if approval identity, Managed Secret reference,
retained executor evidence, exact OpenEvolve pin, ResourceBroker GPU
equivalence, contract, evaluator/data/code identity, or any durable store path
does not match the reviewed A4 envelope.

## Offline launch gate

`test_standard_runtime_native_a4_like_start_resume_and_reuse_v2` uses the same
dependency and graph assembly. It proves multiple generations, population and
archive retention, three simulated GPU leases, checkpointed standard resume,
semantic duplicate suppression, safe feedback, and authoritative
evaluation-reuse-v2 validation without a paid call, GPU, FeTA data, or live
secret.
