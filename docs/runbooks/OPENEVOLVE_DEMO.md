# Offline OpenEvolve demonstrations

The synthetic demonstration uses `examples/tasks/synthetic/openevolve.yaml`, mock agents, deterministic full-file replacements, the synthetic evaluator, and the existing verifier. It progresses from the linear seed (0.78) through tree (0.84) to neural (0.88), then stops at the objective threshold. It makes no model, network, Aura, or patient-data calls.

Example:

```shell
auto-researcher run start \
  --task synthetic \
  --task-config examples/tasks/synthetic/openevolve.yaml \
  --run-id demo-openevolve-001 \
  --thread-id demo-openevolve-thread-001 \
  --max-cycles 4 \
  --checkpoint-db /tmp/demo-openevolve-checkpoints.sqlite \
  --provenance-db /tmp/demo-openevolve-provenance.sqlite \
  --agent-calls-db /tmp/demo-openevolve-agent-calls.sqlite \
  --knowledge-retrievals-db /tmp/demo-openevolve-knowledge.sqlite
```

The integration suite also supplies a fake cell-biology-shaped task component. It evolves a bounded scoring rule over synthetic non-patient signals, then uses the normal synthetic evaluator and verifier. This is a contract compatibility demonstration only: it contains no iCCA execution, biomedical records, Aura access, or new scientific semantics.

Run both, resume equivalence, reconstruction, replay, artefact, and hostile-source checks with:

```shell
pytest -q tests/unit/test_openevolve_contracts.py tests/unit/test_openevolve_sandbox.py tests/integration/test_openevolve_graph.py
```
