# Checkpoint 05A-C prerequisite

Checkpoint 05A and 05A-B identities are consumed and must not be reused. After corrective PR 8.1 merges, create a fresh 05A-C identity and rebuild executor v2 with credential-free Buildx. Bind the checkpoint to the exact Dockerfile, supervisor, child, probe, base image, final image, Docker/BuildKit, host OS and architecture identities.

Repeat every isolation and resource probe. Specifically verify `nr_inodes = file_count_limit + 1`, exact-limit success, limit-plus-one rejection while running, nested-directory accounting, no alternate writable mount, no host output bind, trusted stdout framing, v1/v2 reuse separation, cleanup, transactional evidence, read-only replay and security scanning. Passing PR tests alone does not approve the image.
