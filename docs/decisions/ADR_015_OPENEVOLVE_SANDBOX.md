# ADR 015: Local bounded candidate preparation

Status: accepted for offline PR 6 fixtures.

Candidate preparation uses a fixed framework-owned worker, `python -I -S`, a minimal environment, a fresh private directory, read-only source/input files, a separate writable output directory, a parent-enforced timeout, bounded captured output/logs/file count, and POSIX CPU, address-space, file-size, descriptor, and process limits where supported. It never invokes a candidate-selected command or shell. Static validation rejects filesystem, process, network, reflection, dynamic import, framework import, recursion, unbounded loops, and input mutation primitives before execution. Logs remove tracebacks and absolute paths, and temporary files are deleted.

This is defense in depth for tightly constrained offline fixtures, not a kernel security sandbox. File permissions may be weaker when the process shares the operator UID; POSIX limits vary by platform; static analysis cannot prove arbitrary Python safe; and there is no namespace, seccomp, container, VM, or kernel-enforced network deny. Therefore untrusted live model output must not run under this policy. A later production execution backend must provide an independently verified no-network container or micro-VM boundary, immutable mounts, a dedicated UID, syscall filtering, cgroups, and equivalent safe result transport before live use.
