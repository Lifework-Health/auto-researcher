# Hardened executor threat model

Candidate code is untrusted. Threats include network/DNS/metadata access, host and credential reads, socket/mount/device escape, privilege escalation, process/memory/CPU/output/file exhaustion and unsafe diagnostics. Static PR 6 validation remains mandatory. Docker isolation supplies an independent boundary; the runtime policy binds version, image/base digests, build/entrypoint hashes, mounts, network, privileges, resources and environment.

Residual risks include Docker/runtime/kernel vulnerabilities, daemon privilege, architecture-specific images, macOS VM implementation details and incomplete denial-of-service controls. The real isolation probe is mandatory for live eligibility. Mocks or command inspection cannot approve an image.
