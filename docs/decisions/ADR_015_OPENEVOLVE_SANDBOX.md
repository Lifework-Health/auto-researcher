# ADR 015: Local bounded candidate preparation

Status: accepted for offline PR 6 fixtures.

Candidate preparation uses a fixed framework-owned worker, `python -I -S`, a minimal environment, a fresh private candidate-writable directory, read-only source/input files, a trusted result-transport directory outside that workspace, a parent-enforced timeout, bounded captured output/logs, and POSIX limits where supported. File-count accounting includes only descendants of the candidate-writable directory: regular files, directories, links and special entries. Immutable input, immutable source, trusted transport and framework parent directories are excluded. It remains post-run accounting, not kernel enforcement.

This is defense in depth for tightly constrained offline fixtures, not a kernel security sandbox. File permissions may be weaker when the process shares the operator UID; POSIX limits vary by platform; static analysis cannot prove arbitrary Python safe; and there is no namespace, seccomp, container, VM, or kernel-enforced network deny. Therefore untrusted live model output must not run under this policy. A later production execution backend must provide an independently verified no-network container or micro-VM boundary, immutable mounts, a dedicated UID, syscall filtering, cgroups, and equivalent safe result transport before live use.
