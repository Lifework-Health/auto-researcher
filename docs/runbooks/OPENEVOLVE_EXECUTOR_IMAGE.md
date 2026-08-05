# Build and verify the hardened image

```shell
docker build --provenance=false --sbom=false -t auto-researcher-openevolve-executor:pr7 docker/openevolve-executor
docker image inspect auto-researcher-openevolve-executor:pr7 --format '{{.Id}}'
```

Compare the Dockerfile base digest and compute the recipe/worker hashes. Set `AUTO_RESEARCHER_HARDENED_IMAGE` and `AUTO_RESEARCHER_HARDENED_IMAGE_DIGEST` to the reviewed values, then run the `hardened_executor` tests. A tag alone is never approval. Rebuilds, runtime upgrades and architecture changes require a new reviewed digest and isolation result.
