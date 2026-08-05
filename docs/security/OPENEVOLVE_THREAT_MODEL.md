# OpenEvolve threat model

The candidate source and mutation response are untrusted. Protected assets include credentials, patient or other scientific data, repository and framework code, evaluator/verifier implementations, contracts and budgets, datasets/splits, checkpoints, provenance, reuse records, and artefacts from other candidates.

Threats include command/process creation, network access, filesystem traversal and symlink escape, credential/environment reads, core imports and monkey-patching, reflection/dynamic execution, denial of service, excessive output/files, mutable-input attacks, forged identities, tampered bundles, duplicate work, and sensitive error disclosure.

Controls are a task-declared one-file surface; canonical identities; fail-closed AST and interface checks; no candidate-selected dependencies or commands; isolated interpreter flags and minimal environment; fresh read-only input plus separate output roots; resource/time/output limits; schema validation; deterministic graph decisions; existing evaluator/verifier identity binding; transactional hash-verified artefacts; replay-safe semantic provenance; and sanitized errors/logs. Hostile fixtures exercise each threat category without deliberately launching fork bombs or other genuinely dangerous payloads.

Residual risk remains because the local runner is not a kernel security boundary and network denial is structural rather than namespace-enforced. It is approved only for trusted deterministic/fake offline fixtures. Live or materially untrusted mutations require a hardened executor described in ADR 015.
