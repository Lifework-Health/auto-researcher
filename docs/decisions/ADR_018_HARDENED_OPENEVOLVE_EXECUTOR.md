# ADR 018: Digest-bound Docker executor v2

Status: accepted in code; a fresh checkpoint 05A-C must approve one exact image and platform before live use.

`openevolve-hardened-executor-v2` uses a fixed image-owned supervisor and candidate-child wrapper. The host mounts only immutable input. The sole candidate-writable filesystem is a private `/workspace` tmpfs with `rw,noexec,nosuid,nodev`, UID/GID 65532, a byte ceiling, and `nr_inodes = file_count_limit + 1`; the extra inode is only the mount root. `/tmp`, HOME, host output, repository, credentials, control sockets, evaluator, verifier, and datasets are not mounted writable.

The inode ceiling limits concurrent objects while code runs. It is necessary but insufficient because hard links can create multiple names for one inode. Static validation therefore rejects link, symlink, FIFO, device, and Unix-socket creation, and the trusted supervisor recursively counts every descendant and rejects non-regular/non-directory entries. A deleted entry releases quota; lifetime creation count is not the contract.

Candidate code runs in a child interpreter and cannot frame the result. A dedicated pipe carries bounded structured JSON to the supervisor; candidate stdout and stderr are drained into capped buffers. The supervisor validates identities, finite JSON, workspace population, file types, and hashes, then emits one `openevolve-hardened-worker-result-v2` envelope. Host code accepts exactly one envelope and binds it into `candidate-preparation-v2` evidence.

The base remains `python:3.12.11-slim-bookworm@sha256:519591d6871b7bc437060736b9f7456b8731f1499a57e22e6c285135ae657bf7`. Image, supervisor, child, recipe, runtime, workspace policy, and execution request identities are immutable inputs. Executor-v1 preparation is never accepted as v2. No local-runner fallback is allowed.
