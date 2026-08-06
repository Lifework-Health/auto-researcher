# Checkpoint 05A-D prerequisite

Checkpoint 05A, 05A-B and the failed 05A-C identity are consumed and must not be reused. After corrective PR 8.2 merges, create a fresh 05A-D identity and rebuild executor v2 with credential-free Buildx. Bind the checkpoint to the exact Dockerfile, typed supervisor, child, probe, base image, final image, Docker/BuildKit, host OS and architecture identities.

Repeat every isolation and resource probe. Specifically verify `nr_inodes = file_count_limit + 1`, exact-limit success, limit-plus-one rejection while running, nested-directory accounting, no alternate writable mount, no host output bind, trusted stdout framing, v1/v2 reuse separation, cleanup, transactional evidence, read-only replay and security scanning. Passing PR tests alone does not approve the image.

The trusted supervisor must also pass `mypy docker/openevolve-executor/worker.py`. Its stdout, stderr and control readers each own a typed binary capture with a separate truncation flag, continue draining beyond retention limits, and are consumed only after their threads join. These capture changes do not alter executor v2 policy or worker protocol semantics.
