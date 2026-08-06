# Hardened executor platform matrix

| Platform | Status | Evidence |
|---|---|---|
| Docker Desktop 29.3.1, Linux VM, arm64 | PR 8.1 integration-tested; live approval pending 05A-C | Real v2 inode, resource, network, mount, environment and preparation tests must pass for the final digest |
| Linux Docker/OCI, other versions or architectures | Integration-gated | Build, pin and run the real isolation gate |
| macOS without Docker Linux engine | Unavailable | No native kernel sandbox implemented |
| Windows, Podman, bubblewrap, nsjail, firejail, gVisor | Unavailable | Not present and not tested for executor v2 |

The PR 6 local runner remains test-only for trusted deterministic fixtures and is never live-mutation eligible.
