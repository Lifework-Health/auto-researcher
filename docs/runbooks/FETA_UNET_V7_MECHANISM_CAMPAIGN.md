# FeTA U-Net V7 mechanism campaign

V7 is a separately identified fold-0 development campaign. It does not inspect
the sealed holdout and does not seed from the external approximately 0.84 U-Net.
It searches MONAI DynUNet mechanisms inside a 15M-150M trainable-parameter
envelope and a 44 GiB measured peak-allocation ceiling.

## Frozen campaign shape

- Four deterministic 25-epoch roots vary depth/stage allocation, receptive
  field, residual blocks and deep supervision.
- Each root can produce three novel OpenEvolve structural mutations. Generation
  zero reuses the verified root and is not retrained.
- The strongest four structurally distinct parents receive two lineage-local
  Optuna trials each. Those trials hold architecture fixed and tune learning
  rate, weight decay, dropout and Dice weight.
- Two controlled DIRECT wildcards provide bounded mechanism/objective escapes.
- The maximum promotion ladder is 8 to epoch 50, 4 to epoch 100 and 2 to epoch
  150. It is a ceiling, not a promise: completion-aware scheduling stops new
  exploration with 5.5 hours remaining and protects finalist graduation before
  the 22-hour hard deadline.

## Fail-closed launch order

1. Check out the reviewed V7 commit in a clean, dedicated worktree and install
   its locked environment.
2. Copy `campaign-22h-v7-template.yaml` and `contract-22h-v7.yaml` into a fresh
   protected runtime directory. Replace every absolute-path placeholder. Do not
   create control databases or call a model yet.
3. Run the static preflight:

   ```bash
   PYTHONPATH="$V7_REPO/src" "$V7_PY" -m \
     auto_researcher.tasks.feta_unet_search.v7_preflight \
     --mode static \
     --task-config "$V7_CONFIG" \
     --contract "$V7_CONTRACT"
   ```

   It must report four unique roots, the exact portfolio and contract identities,
   15M-150M parameters, a 44 GiB ceiling and `model_calls_performed: 0`.

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

5. Only after both gates pass, make a fresh control directory, resolve the live
   model credential in the launch shell, and start the campaign under `nohup`.
   Keep the PID, launcher log, configuration, contract, static report and CUDA
   report in the protected runtime root.

## Runtime interpretation

The 22-hour deadline consists of a 20-hour nominal campaign and up to two hours
of graduation-only grace. A block is admitted only when its conservative
duration plus the appropriate reserve fits. When exploration no longer fits,
the controller converts the request into a DIRECT continuation of the strongest
diverse unfinished finalist to epoch 150. The final 30 minutes remain reserved
for durable results and reporting.

OpenEvolve candidates must change at least one structural mechanism relative to
their parent; training-only mutations are rejected. Local Optuna branches do the
opposite: they preserve the exact evolved architecture and vary only the four
registered continuous optimisation parameters. Candidate parameter count and
observed GPU peak are verified again during execution.
