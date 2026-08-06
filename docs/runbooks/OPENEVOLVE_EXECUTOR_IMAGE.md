# Build and verify the hardened executor v2 image

Use a credential-free Docker CLI configuration that contains only the verified Docker Desktop Buildx plugin and `{ "auths": {} }`. Do not copy the host Docker configuration, pass SSH, use build secrets, or push.

```shell
DOCKER_CONFIG=<credential-free-config> BUILDX_CONFIG=<isolated-buildx-state> \
docker buildx build --platform linux/arm64 --provenance=false --sbom=false \
  --load --pull --no-cache -f docker/openevolve-executor/Dockerfile \
  -t auto-researcher-openevolve-executor:pr81 docker/openevolve-executor
docker image inspect auto-researcher-openevolve-executor:pr81 --format '{{.Id}}'
```

Record Dockerfile, supervisor, candidate-child, isolation-probe and build-context hashes. Bind tests to the returned image ID, never the tag alone. Run the real `hardened_executor` suite and verify `/workspace` tmpfs options, exact inode ceiling, eighth/ninth entry behavior, absence of a writable host-output mount, supervisor framing, resource limits and cleanup. Build timestamps may change local image/layer digests, so do not claim bit-for-bit reproducibility without matching evidence.

When configuring `HardenedDockerExecutor(workspace_root=...)`, use a dedicated trusted runtime parent—not the repository, artefact output, approval directory, or a candidate-controlled path. The executor may materialise a missing nested parent with private host permissions and retains it across operations and resume. It rejects a final symlink, non-directory, or inaccessible parent with `hardened_executor_workspace_root_unavailable`. Only random per-operation children are removed. The trusted host parent is never exposed as a writable container mount; Docker receives only the per-operation immutable input bind and the private `/workspace` tmpfs.

This build does not approve live use. A fresh checkpoint 05A-C must validate one exact digest on the intended runtime/platform.
