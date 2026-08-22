# FeTA U-Net V6 finalist extension

This runbook continues two verified V6 trajectories from epoch 100 to epoch 150.
It does not restart training, call an agent model, access the holdout partition, or
change the completed V6 result bundles.

The extension restores the exact epoch-100 model, optimiser, scheduler, AMP
scaler, RNG and data-order state. Existing best checkpoints and validation
history are copied into deterministic extension experiment directories before
epochs 101-150 are executed.

## Selected finalists

- `experiment-b1172a7397a5bf78`: peak-performing `v6_deep_80` trajectory,
  best validation Dice `0.8199315689827265` at epoch 90.
- `experiment-4f83282470f00fd3`: stable `v6_balanced_96` trajectory,
  validation Dice `0.8178605150848705` at epoch 100.

Both source artefact bundles, continuation manifests and checkpoint hashes must
pass the action-bound preflight. The two trajectory identities must be distinct.

## Read-only preflight

Run from `SBIGPUServer1` after installing the reviewed extension commit in the
V6 worktree:

```bash
V6_REPO="$HOME/projects/auto_researcherv2.1-unet-v6"
V6_PY="$V6_REPO/.venv/bin/python"

V6_ROOT="/mnt/R0/Projects/POIAZ/gmorgan/auto-researcher-runtime/feta-unet/campaign-20h-v6-v3-20260820"
V6_RUN_ID="feta-unet-campaign-20h-v6-v3-20260820"

EXT_ROOT="$V6_ROOT/finalist-extension-150-20260822"
EXT_RUN_ID="feta-unet-v6-finalist-extension-150-20260822"

PYTHONPATH="$V6_REPO/src" "$V6_PY" -m \
  auto_researcher.tasks.feta_unet_search.finalist_extension \
  --mode preflight \
  --runtime-root "$V6_ROOT" \
  --source-run-id "$V6_RUN_ID" \
  --extension-root "$EXT_ROOT" \
  --extension-run-id "$EXT_RUN_ID" \
  --experiment-id experiment-b1172a7397a5bf78 \
  --experiment-id experiment-4f83282470f00fd3
```

Confirm that the preflight reports two candidates, source fidelity 100, target
fidelity 150, two distinct trajectory identities and a bounded estimated wall
time. Preflight performs no writes and no CUDA training.

Before launch, GPU 1 must have no compute process:

```bash
nvidia-smi -i 1 \
  --query-compute-apps=pid,process_name,used_memory \
  --format=csv,noheader
```

## Unattended extension launch

The launch uses a five-hour wall-time admission budget. It checks the remaining
budget before starting each finalist. A completed first result remains durable
if the second finalist cannot be admitted.

```bash
if [[ -e "$EXT_ROOT" ]]; then
  echo "STOP: V6 finalist extension root already exists"
elif nvidia-smi -i 1 \
  --query-compute-apps=pid \
  --format=csv,noheader,nounits 2>/dev/null | grep -q '[0-9]'; then
  echo "STOP: GPU 1 has a compute process"
else
  umask 077
  install -d -m 700 "$EXT_ROOT/logs"

  CUDA_VISIBLE_DEVICES=1 nohup \
    env PYTHONPATH="$V6_REPO/src" \
    "$V6_PY" -u -m \
    auto_researcher.tasks.feta_unet_search.finalist_extension \
    --mode run \
    --runtime-root "$V6_ROOT" \
    --source-run-id "$V6_RUN_ID" \
    --extension-root "$EXT_ROOT" \
    --extension-run-id "$EXT_RUN_ID" \
    --task-config "$V6_ROOT/config/campaign-20h.yaml" \
    --contract "$V6_ROOT/config/contract-20h.yaml" \
    --maximum-wall-time-seconds 18000 \
    --experiment-id experiment-b1172a7397a5bf78 \
    --experiment-id experiment-4f83282470f00fd3 \
    >"$EXT_ROOT/logs/launcher.log" 2>&1 &

  EXT_PID=$!
  printf '%s\n' "$EXT_PID" >"$EXT_ROOT/logs/launcher.pid"
  chmod 600 "$EXT_ROOT/logs/launcher.pid"
  echo "V6_FINALIST_EXTENSION_LAUNCHED pid=$EXT_PID gpu=1"
fi
```

## Monitoring

```bash
EXT_PID="$(cat "$EXT_ROOT/logs/launcher.pid")"

if kill -0 "$EXT_PID" 2>/dev/null; then
  echo "V6_FINALIST_EXTENSION_RUNNING"
  ps -o pid,etime,stat,cmd -p "$EXT_PID"
else
  echo "V6_FINALIST_EXTENSION_EXITED"
fi

grep -E \
  'FETA_UNET_PROMOTION_RESUMED|FETA_UNET_PROGRESS|FETA_UNET_AMP|failed|CUDA out of memory' \
  "$EXT_ROOT/logs/launcher.log" | tail -n 120

nvidia-smi
```

Each candidate must emit `FETA_UNET_PROMOTION_RESUMED` with `from_epoch=100`
and `to_epoch=150`. A run that begins at epoch 1 is invalid and must be stopped.

## V7 handoff

After both candidates complete, these protected extension artefacts are written:

- `extension-plan.json`: frozen source identities, target configurations and
  measured time estimates.
- `extension-summary.json`: best and endpoint scores, epochs, duration, memory
  and resume evidence.
- `v7-seed-evidence.json`: ranked V7 parent records, safe initial observations,
  the exact incumbent configuration and 25-epoch root forms.

Verify readiness without training:

```bash
"$V6_PY" - "$EXT_ROOT/v7-seed-evidence.json" <<'PY'
import json
import sys
from pathlib import Path

data = json.loads(Path(sys.argv[1]).read_text())
print("schema_version:", data["schema_version"])
print("ready:", data["ready"])
print("parent_count:", len(data["parent_candidates"]))
for parent in data["parent_candidates"]:
    print({
        "experiment": parent["extension_experiment_id"],
        "trajectory": parent["trajectory_identity"],
        "best_score": parent["best_score"],
        "best_epoch": parent["best_epoch"],
        "endpoint_score": parent["endpoint_score"],
        "peak_gpu_memory_bytes": parent["peak_gpu_memory_bytes"],
    })
PY
```

V7 preparation must require `ready: true` and consume this file by exact
content hash. Do not manually transcribe the finalist scores or configurations.
