# Hardened executor platform matrix

| Platform | Status | Evidence |
|---|---|---|
| Docker Desktop 29.3.1, Linux VM, arm64 | Fully supported for PR 7 offline gate | Real digest, network, mount, environment and preparation tests passed |
| Linux Docker/OCI, other versions or architectures | Integration-gated | Build, pin and run the real isolation gate |
| macOS without Docker Linux engine | Unavailable | No native kernel sandbox implemented |
| Windows, Podman, bubblewrap, nsjail, firejail, gVisor | Unavailable | Not present and not tested in PR 7 |

The PR 6 local runner remains test-only for trusted deterministic fixtures and is never live-mutation eligible.
