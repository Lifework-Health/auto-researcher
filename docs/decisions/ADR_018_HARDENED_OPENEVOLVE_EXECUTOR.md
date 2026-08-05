# ADR 018: Digest-bound Docker executor

Status: accepted on tested Docker Desktop Linux engines; unavailable elsewhere unless the real gate passes.

The executor uses a fixed image and entrypoint, `--network none`, read-only root, UID/GID 65532, all capabilities dropped, no-new-privileges, PID/memory/CPU/time limits, bounded tmpfs, one read-only input mount, one writable output mount and three deterministic locale/time environment values. It never mounts the repository, home, Docker socket, evaluator, verifier or datasets. Host-side code validates strict bounded JSON before task-owned `ExperimentSpec` conversion.

The base is `python:3.12.11-slim-bookworm@sha256:519591d6871b7bc437060736b9f7456b8731f1499a57e22e6c285135ae657bf7`. The locally verified linux/arm64 image was `sha256:a7fc72f30d80a5d736f022121c7e3ba512454e38492b289666c92cd1e1b3d7a9`; other architectures must build and explicitly approve their own digest. No silent local-runner fallback is allowed.
