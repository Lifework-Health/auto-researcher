# FeTA V10 preparation

V10 is a 36-hour, fold-0-only DynUNet mechanism campaign. It does not access the sealed holdout. Its scientific aim is to improve external-CSF and cortical-grey-matter segmentation without changing the primary macro-Dice objective.

## Frozen envelope

- Six fixed 30-epoch Direct roots cover three architecture profiles and three registered loss/sampling mechanism combinations.
- Four verified roots receive two local Optuna trials each.
- OpenEvolve must produce six novel bounded DynUNet children.
- Twenty unique 30-epoch trajectories graduate 10 to epoch 50, six to epoch 100 and four to epoch 150.
- The last eight hours are protected for graduation and evidence finalisation.

The Research Director uses Opus with xhigh effort and may emit up to 20 valid typed strategy decisions. It owns scientific prioritisation only. The deterministic campaign-portfolio compiler owns exact search allocations and produces executable requests, so routine Planner structured-output failures cannot alter or stop the frozen schedule. The Supervisor continues to enforce the contract, resource, evidence and deadline gates.

## Literature boundary

The Literature Scout output is retrieved and reviewed before launch, then frozen in `v10-bound-evidence.json`. External text is untrusted advisory evidence: it may motivate a typed directive but cannot construct, approve or execute an experiment. Live mutable literature search is prohibited during this campaign.

## Preparation check

Run the offline preflight from the exact V10 checkout. It performs no model calls, GPU work or holdout access:

```bash
PYTHONPATH=src python -m auto_researcher.tasks.feta_unet_search.v10_preflight \
  --config examples/tasks/feta_unet_search/campaign-36h-v10-template.yaml \
  --contract examples/tasks/feta_unet_search/contract-36h-v10.yaml \
  --evidence examples/tasks/feta_unet_search/v10-bound-evidence.json
```

`launch_ready: false` is expected at this stage. The real-CUDA mechanism smoke is bound in `v10-cuda-mechanism-smoke.json`; its generalized-Dice-focal AMP step and weak-tissue crop both passed on an RTX A6000 without dataset or holdout access. Do not launch until the remaining action-bound preflight passes on the exact clean server checkout.
