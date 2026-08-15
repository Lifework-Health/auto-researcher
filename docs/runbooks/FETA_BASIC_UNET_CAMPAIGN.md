# Planner-driven FeTA BasicUNet campaign

This is a development campaign for `feta_unet_search@1.0`. It is intentionally
lighter than the hardened A4 operator flow: it runs directly in the project
environment on one selected GPU and uses the normal Auto Researcher planner.

## Fixed scientific envelope

- FeTA development fold 0 only; the holdout remains sealed.
- MONAI 3D BasicUNet with features `(32, 32, 64, 128, 256, 32)`.
- The existing RAS, 0.5 mm, foreground crop, nonzero z-score and `128^3`
  patch pipeline.
- One visible GPU and no distributed training.
- Subject-level macro Dice over labels 1-7 is the primary score.

The planner may allocate each block to DIRECT, Optuna or OpenEvolve. It may
vary only learning rate, weight decay, dropout, Dice loss weight,
positive/negative patch ratio, augmentation strength and the registered
5/25/100/150-epoch fidelity. OpenEvolve mutates a small JSON training policy;
it cannot mutate the network or execute the trusted evaluator as candidate
code.

## Before the campaign

1. Let the 150-epoch fold-0 baseline finish and retain its result and
   milestone checkpoints.
2. Compute `campaign_seconds_per_epoch` from its reported training duration.
   Use `training_duration_seconds / 150`, rounded up by about 15 percent. This
   conservative estimate is used to refuse a proposed block that would not fit
   before finalisation.
3. Copy `campaign-20h-template.yaml` outside the repository. Replace the four
   absolute-path placeholders and the seconds-per-epoch estimate. Keep
   `CUDA_VISIBLE_DEVICES: "0"`.
4. Create a fresh runtime root with separate `control`, `output` and
   `workspace` directories. Reusing the baseline preprocessing cache is
   optional; do not reuse its control databases.
5. Export `ANTHROPIC_API_KEY` without writing it into YAML or a log.

## End-to-end smoke

Before committing 20 hours, make a smoke copy of the contract and task config:

- set `campaign_duration_seconds` to `3600`;
- set `campaign_finalisation_reserve_seconds` to `300` in both files;
- set `maximum_cycles` to `1` and `maximum_experiments` to `1`;
- set `maximum_total_model_calls` to `4`;
- keep `maximum_epochs: 5`, `openevolve_fidelity: 5` and one visible GPU.

Launch the smoke through the standard `run start` command with fresh SQLite
files. It passes only if a live planner call succeeds, one real fold-0
candidate trains and validates, evidence is recorded, and the run reaches a
terminal state without touching the holdout.

## 20-hour launch

The reviewed template and contract use a 72,000-second campaign deadline with
a 1,800-second finalisation reserve, at most 12 planner cycles and at most 30
real candidate evaluations. Use a fresh run ID and thread ID:

```bash
CAMPAIGN_ROOT=/absolute/path/to/runtime
UNET_REPO=/absolute/path/to/auto_researcherv2.1-unet-staging

CUDA_VISIBLE_DEVICES=0 nohup "$UNET_REPO/.venv/bin/auto-researcher" run start \
  --task feta_unet_search \
  --contract "$UNET_REPO/examples/tasks/feta_unet_search/contract-20h.yaml" \
  --task-config "$CAMPAIGN_ROOT/config/campaign-20h.yaml" \
  --run-id feta-unet-campaign-20h-20260816 \
  --thread-id feta-unet-campaign-20h-20260816 \
  --checkpoint-db "$CAMPAIGN_ROOT/control/checkpoints.sqlite" \
  --provenance-db "$CAMPAIGN_ROOT/control/provenance.sqlite" \
  --optuna-db "$CAMPAIGN_ROOT/control/optuna.sqlite" \
  --agent-calls-db "$CAMPAIGN_ROOT/control/agent-calls.sqlite" \
  --knowledge-retrievals-db "$CAMPAIGN_ROOT/control/knowledge.sqlite" \
  >"$CAMPAIGN_ROOT/campaign.log" 2>&1 </dev/null &
```

Record the launcher PID immediately. The 20-hour clock begins when
`initialise_run` executes. A candidate already admitted may finish, but no new
Optuna or OpenEvolve candidate is admitted after the deadline. The conservative
pre-admission estimate and reserve are what keep normal completion inside the
intended window.

## Monitoring and final comparison

Monitor the launcher, recent log lines, GPU utilisation, checkpoint database
growth and aggregate result JSON. Do not infer failure solely from brief idle
GPU periods while the planner or validation code is running.

At completion compare every verified candidate with the frozen 25-, 100- and
150-epoch baseline milestones at matched fidelity. Report the best feasible
configuration, subject-level macro Dice, per-tissue Dice, reconstruction gap,
best epoch, training duration and search method. Treat fold-0 gains as
development evidence, not final generalisation evidence; confirmation on
additional fixed folds is a separate follow-up run.
