# FeTA U-Net V7 mechanism campaign

V7 is a separately identified fold-0 development campaign. It does not inspect
the sealed holdout and does not seed from the external approximately 0.84 U-Net.
It searches a constrained structural BasicUNet grammar inside a 15M-150M
trainable-parameter envelope and a 44 GiB measured peak-allocation ceiling.
Every candidate retains a recognisable BasicUNet encoder-skip-decoder lineage;
V7 does not switch to DynUNet.

## Frozen campaign shape

- Four deterministic 25-epoch roots vary depth/stage allocation, convolutions
  per stage, receptive field and dilation, residual blocks, skip fusion,
  down/up operators and deep supervision.
- Each root can produce three novel OpenEvolve structural mutations. Generation
  zero reuses the verified root and is not retrained.
- The two verified V6 150-epoch finalists are bound into mutation context with
  their exact configurations and scores. They are evidence parents only and
  are not retrained as V7 screening candidates.
- The public-safe REQ-11 panel identity, aggregate candidate deltas,
  complementarity counts and diagnostic priorities are validated as immutable
  portfolio input and passed to every OpenEvolve mutation. They guide mechanism
  selection but do not alter the primary macro-Dice objective.
- The strongest four structurally distinct parents receive two lineage-local
  Optuna trials each. Those trials hold architecture fixed and tune learning
  rate, weight decay, dropout, Dice weight, sampling ratio, loss family,
  augmentation policy and schedule. These eight trials are the reserved
  data/objective lane (8 of the 22 post-root exploration candidates, over 25%).
- Two controlled DIRECT wildcards provide bounded mechanism/objective escapes.
- The maximum promotion ladder is 8 to epoch 50, 4 to epoch 100 and 2 to epoch
  150. Completion-aware scheduling stops new exploration with 6.75 hours
  remaining. That reserve is mechanically checked against two worst-case
  25-to-150 continuations at 90 seconds/epoch plus 30 minutes of reporting.

## Fail-closed launch order

1. Check out the reviewed V7 commit in a clean, dedicated worktree and install
   its locked environment.
2. Copy `campaign-22h-v7-template.yaml` and `contract-22h-v7.yaml` into a fresh
   protected runtime directory. Replace every absolute-path placeholder. Do not
   create control databases or call a model yet. The campaign template must use
   an `experiment:` section containing the first structural root and finite
   OpenEvolve controls. A `search: {type: DIRECT}` section is invalid and the
   static gate rejects it.
3. Run the static preflight:

   ```bash
   PYTHONPATH="$V7_REPO/src" "$V7_PY" -m \
     auto_researcher.tasks.feta_unet_search.v7_preflight \
     --mode static \
     --task-config "$V7_CONFIG" \
     --contract "$V7_CONTRACT"
   ```

   It must report `initial_search_type: DIRECT`, four unique roots, the exact
   portfolio and contract identities, the bound REQ-11 panel, 15M-150M
   parameters, a 44 GiB ceiling and `model_calls_performed: 0`.

4. Confirm the selected A6000 is idle. Expose exactly that one device and run the
   real-CUDA gate before credentials are entered or campaign state is created:

   ```bash
   CUDA_VISIBLE_DEVICES="$V7_GPU" PYTHONPATH="$V7_REPO/src" "$V7_PY" -m \
     auto_researcher.tasks.feta_unet_search.v7_preflight \
     --mode cuda \
     --task-config "$V7_CONFIG" \
     --contract "$V7_CONTRACT" \
     >"$V7_PREFLIGHT_LOG" 2>&1
   ```

   All four roots must complete one real AMP forward/backward/optimizer step.
   Each measured peak must be at or below 47,244,640,256 bytes. Any occupied
   GPU, OOM, non-finite output/loss, identity mismatch or ceiling breach is a
   hard `PRE-RUN BLOCKED` condition.

5. Run the adaptive-backend capability gate with disposable SQLite stores. V7
   begins with DIRECT roots but later routes lineage-local Optuna studies, so
   the locked runtime must expose the Optuna backend before durable campaign
   state is created. The production launch passes the dedicated Optuna store
   even though the configured initial search type is DIRECT.

6. Only after all gates pass, make a fresh control directory, resolve the live
   model credential in the launch shell, and start the campaign under `nohup`.
   Keep the PID, launcher log, configuration, contract, static report and CUDA
   report in the protected runtime root.

## Runtime interpretation

The 22-hour deadline consists of an exploration phase and a protected
graduation/reporting phase. A block is admitted only when its conservative
duration plus the appropriate reserve fits. When exploration no longer fits,
the controller converts the request into a DIRECT continuation of the strongest
diverse unfinished finalist to epoch 150. The second graduation preferentially
uses a different root lineage. The final 30 minutes remain reserved for durable
results and reporting.

After the two finalists complete, a separate inference-calibration lane may
compare bounded overlap, Gaussian/constant blending and flip-TTA variants.
Class-specific postprocessing is diagnostic-gated. Calibration results are kept
separate from training gains and do not alter the training-search identity.

OpenEvolve candidates must change at least one structural mechanism relative to
their parent; training-only mutations are rejected. Local Optuna branches do the
opposite: they preserve the exact evolved architecture and vary learning rate,
weight decay, dropout, Dice weight, sampling ratio and schedule. Candidate
parameter count and observed GPU peak are verified again during execution.
