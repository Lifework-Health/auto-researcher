"""Resource-limited local runner with a sanitised process boundary."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from auto_researcher.search.openevolve.models import (
    CandidateExecutionStatus,
    CandidatePreparationResult,
    CandidateValidationStatus,
    EvolvableComponentSpec,
    OpenEvolveCandidate,
    SandboxPolicy,
)


def _resource_limits(policy: SandboxPolicy):
    def apply() -> None:
        try:
            import resource

            resource.setrlimit(
                resource.RLIMIT_CPU, (policy.cpu_time_seconds, policy.cpu_time_seconds)
            )
            resource.setrlimit(
                resource.RLIMIT_AS, (policy.memory_bytes, policy.memory_bytes)
            )
            resource.setrlimit(
                resource.RLIMIT_FSIZE, (policy.output_bytes, policy.output_bytes)
            )
            resource.setrlimit(
                resource.RLIMIT_NOFILE,
                (policy.file_count_limit + 8, policy.file_count_limit + 8),
            )
            if hasattr(resource, "RLIMIT_NPROC"):
                resource.setrlimit(
                    resource.RLIMIT_NPROC, (policy.process_limit, policy.process_limit)
                )
        except (OSError, ValueError):
            # Platform support is best-effort; the parent timeout/output caps remain authoritative.
            pass

    return apply


def _safe_log(stdout: bytes, stderr: bytes, limit: int) -> tuple[str, bool]:
    raw = (stdout + b"\n" + stderr).decode("utf-8", errors="replace")
    raw = re.sub(
        r"Traceback \(most recent call last\):.*",
        "candidate_execution_failed",
        raw,
        flags=re.S,
    )
    raw = re.sub(r"/(?:Users|home|private|tmp|var)/[^\s:]+", "<redacted-path>", raw)
    encoded = raw.encode("utf-8")
    truncated = len(encoded) > limit
    return encoded[:limit].decode("utf-8", errors="ignore").strip(), truncated


class LocalSandboxRunner:
    """A strong local boundary, but not a kernel security sandbox."""

    runner_id = "local-resource-limited-runner-v1"

    def __init__(self, workspace_root: Path | None = None) -> None:
        self.workspace_root = workspace_root

    def prepare(
        self,
        candidate: OpenEvolveCandidate,
        component: EvolvableComponentSpec,
        policy: SandboxPolicy,
        configuration: dict,
    ) -> CandidatePreparationResult:
        started = time.monotonic()
        temporary: Path | None = None
        try:
            base = self.workspace_root
            if base is not None:
                base.mkdir(parents=True, exist_ok=True)
            temporary = Path(tempfile.mkdtemp(prefix="openevolve-candidate-", dir=base))
            immutable = temporary / "immutable"
            writable = temporary / "candidate-output"
            immutable.mkdir(mode=0o700)
            writable.mkdir(mode=0o700)
            source_path = immutable / component.mutable_file
            input_path = immutable / "input.json"
            output_path = writable / "output.json"
            source_path.write_text(
                candidate.source_payload.replace("\r\n", "\n"),
                encoding="utf-8",
                newline="\n",
            )
            input_path.write_text(
                json.dumps(
                    configuration,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ),
                encoding="utf-8",
            )
            source_path.chmod(0o400)
            input_path.chmod(0o400)
            command = [
                sys.executable,
                "-I",
                "-S",
                str(Path(__file__).with_name("worker.py")),
                str(source_path),
                component.entry_point,
                str(input_path),
                str(output_path),
            ]
            environment = {
                "PYTHONHASHSEED": str(
                    int(
                        hashlib.sha256(
                            f"{candidate.candidate_id}\x1f{policy.policy_id}".encode()
                        ).hexdigest()[:8],
                        16,
                    )
                ),
                "TZ": policy.timezone,
                "LC_ALL": policy.locale,
                "LANG": policy.locale,
            }
            try:
                completed = subprocess.run(
                    command,
                    cwd=writable,
                    env=environment,
                    capture_output=True,
                    timeout=policy.wall_time_seconds,
                    check=False,
                    preexec_fn=_resource_limits(policy) if os.name == "posix" else None,
                )
            except subprocess.TimeoutExpired as exc:
                log, truncated = _safe_log(
                    exc.stdout or b"", exc.stderr or b"", policy.log_bytes
                )
                return CandidatePreparationResult(
                    candidate_id=candidate.candidate_id,
                    validation_status=CandidateValidationStatus.VALID,
                    execution_status=CandidateExecutionStatus.TIMED_OUT,
                    safe_error_code="candidate_timeout",
                    timeout=True,
                    safe_log_excerpt=log,
                    log_truncated=truncated,
                    runtime_seconds=round(time.monotonic() - started, 6),
                    cleanup_complete=True,
                )
            log, truncated = _safe_log(
                completed.stdout, completed.stderr, policy.log_bytes
            )
            if completed.returncode != 0:
                code = (
                    "candidate_resource_limit"
                    if completed.returncode < 0
                    else "candidate_execution_failed"
                )
                return CandidatePreparationResult(
                    candidate_id=candidate.candidate_id,
                    validation_status=CandidateValidationStatus.VALID,
                    execution_status=(
                        CandidateExecutionStatus.RESOURCE_LIMITED
                        if completed.returncode < 0
                        else CandidateExecutionStatus.FAILED
                    ),
                    safe_error_code=code,
                    resource_limited=completed.returncode < 0,
                    safe_log_excerpt=log,
                    log_truncated=truncated,
                    runtime_seconds=round(time.monotonic() - started, 6),
                    cleanup_complete=True,
                )
            if (
                not output_path.is_file()
                or output_path.stat().st_size > policy.output_bytes
            ):
                raise ValueError("candidate_output_limit")
            if len(list(temporary.rglob("*"))) > policy.file_count_limit:
                raise ValueError("candidate_output_limit")
            output_bytes = output_path.read_bytes()
            output = json.loads(output_bytes)
            if not isinstance(output, dict):
                raise ValueError("candidate_experiment_spec_invalid")
            return CandidatePreparationResult(
                candidate_id=candidate.candidate_id,
                validation_status=CandidateValidationStatus.VALID,
                execution_status=CandidateExecutionStatus.COMPLETED,
                output_hashes=(hashlib.sha256(output_bytes).hexdigest(),),
                generated_configuration=output,
                safe_log_excerpt=log,
                log_truncated=truncated,
                runtime_seconds=round(time.monotonic() - started, 6),
                cleanup_complete=True,
            )
        except (OSError, ValueError, json.JSONDecodeError):
            return CandidatePreparationResult(
                candidate_id=candidate.candidate_id,
                validation_status=CandidateValidationStatus.VALID,
                execution_status=CandidateExecutionStatus.FAILED,
                safe_error_code="candidate_execution_failed",
                runtime_seconds=round(time.monotonic() - started, 6),
                cleanup_complete=True,
            )
        finally:
            if temporary is not None and temporary.exists():
                import shutil

                shutil.rmtree(temporary)
