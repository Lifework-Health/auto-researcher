"""Digest-bound Docker executor with independently tested no-network isolation."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from auto_researcher.runtime.identity import payload_hash
from auto_researcher.search.openevolve.models import (
    CandidateExecutionStatus,
    CandidatePreparationResult,
    CandidateValidationStatus,
    EvolvableComponentSpec,
    OpenEvolveCandidate,
    SandboxPolicy,
)
from auto_researcher.search.openevolve.upstream_models import (
    ExecutorIsolationResult,
    HardenedExecutorPolicy,
)


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def docker_policy(
    image_reference: str,
    image_digest: str,
    recipe: Path,
    worker: Path,
    runtime_version: str,
) -> HardenedExecutorPolicy:
    return HardenedExecutorPolicy(
        runtime_version=runtime_version,
        image_reference=image_reference,
        image_digest=image_digest,
        entrypoint_hash=file_hash(worker),
        build_recipe_hash=file_hash(recipe),
    )


class HardenedDockerExecutor:
    runner_id = "openevolve-hardened-executor-v1"

    def __init__(
        self, policy: HardenedExecutorPolicy, workspace_root: Path | None = None
    ):
        self.policy = policy
        self.workspace_root = workspace_root

    def _inspect(self) -> None:
        if shutil.which("docker") is None:
            raise ValueError("hardened_executor_unavailable")
        try:
            version = subprocess.run(
                ["docker", "version", "--format", "{{.Server.Version}}"],
                capture_output=True,
                text=True,
                timeout=5,
                check=True,
            ).stdout.strip()
            image_id = subprocess.run(
                [
                    "docker",
                    "image",
                    "inspect",
                    self.policy.image_reference,
                    "--format",
                    "{{.Id}}",
                ],
                capture_output=True,
                text=True,
                timeout=5,
                check=True,
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError) as exc:
            raise ValueError("hardened_executor_unavailable") from exc
        if version != self.policy.runtime_version:
            raise ValueError("hardened_executor_runtime_mismatch")
        if image_id != self.policy.image_digest:
            raise ValueError("hardened_executor_image_mismatch")

    def _base_command(
        self, input_dir: Path, output_dir: Path, *, entrypoint: str | None = None
    ) -> list[str]:
        command = [
            "docker",
            "run",
            "--rm",
            "--network",
            "none",
            "--read-only",
            "--user",
            "65532:65532",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--pids-limit",
            "16",
            "--memory",
            "256m",
            "--cpus",
            "1",
            "--tmpfs",
            "/tmp:rw,noexec,nosuid,size=16m",
            "--mount",
            f"type=bind,src={input_dir},dst=/input,readonly",
            "--mount",
            f"type=bind,src={output_dir},dst=/output",
            "--env",
            "TZ=UTC",
            "--env",
            "LC_ALL=C",
            "--env",
            "LANG=C",
        ]
        if entrypoint is not None:
            command.extend(["--entrypoint", entrypoint])
        command.append(self.policy.image_reference)
        return command

    @staticmethod
    def _safe_log(output: bytes, limit: int) -> str:
        text = output.decode("utf-8", errors="replace")
        text = re.sub(
            r"/(?:Users|home|private|tmp|var)/[^\s:]+", "<redacted-path>", text
        )
        text = re.sub(
            r"Traceback \(most recent call last\):.*",
            "candidate_execution_failed",
            text,
            flags=re.S,
        )
        return text.encode()[:limit].decode(errors="ignore")

    def verify_isolation(self) -> ExecutorIsolationResult:
        self._inspect()
        root = Path(
            tempfile.mkdtemp(prefix="openevolve-isolation-", dir=self.workspace_root)
        )
        try:
            inputs, outputs = root / "input", root / "output"
            inputs.mkdir()
            outputs.mkdir(mode=0o777)
            outputs.chmod(0o777)
            command = self._base_command(inputs, outputs, entrypoint="python")
            command.append("/opt/runner/isolation_probe.py")
            completed = subprocess.run(
                command, capture_output=True, timeout=10, check=False
            )
            path = outputs / "isolation.json"
            checks = (
                json.loads(path.read_text())
                if completed.returncode == 0 and path.is_file()
                else {}
            )
            network = all(
                checks.get(key) is True
                for key in (
                    "dns_denied",
                    "outbound_tcp_denied",
                    "loopback_denied",
                    "metadata_denied",
                )
            )
            mounts = all(
                checks.get(key) is True
                for key in (
                    "host_sentinel_hidden",
                    "docker_socket_hidden",
                    "ssh_hidden",
                )
            )
            environment = checks.get("credential_names_absent") is True
            return ExecutorIsolationResult(
                executor_policy_hash=payload_hash(self.policy),
                network_isolation_verified=network,
                mount_isolation_verified=mounts,
                environment_sanitisation_verified=environment,
                safe_checks=checks,
                safe_error_code=None
                if network and mounts and environment
                else "hardened_executor_network_isolation_unverified",
            )
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def prepare(
        self,
        candidate: OpenEvolveCandidate,
        component: EvolvableComponentSpec,
        policy: SandboxPolicy,
        configuration: dict,
    ) -> CandidatePreparationResult:
        started = time.monotonic()
        try:
            isolation = self.verify_isolation()
            if not isolation.network_isolation_verified:
                raise ValueError("hardened_executor_network_isolation_unverified")
        except ValueError as exc:
            return CandidatePreparationResult(
                candidate_id=candidate.candidate_id,
                validation_status=CandidateValidationStatus.VALID,
                execution_status=CandidateExecutionStatus.FAILED,
                safe_error_code=str(exc),
                cleanup_complete=True,
            )
        root = Path(
            tempfile.mkdtemp(prefix="openevolve-hardened-", dir=self.workspace_root)
        )
        try:
            inputs, outputs = root / "input", root / "output"
            inputs.mkdir()
            outputs.mkdir(mode=0o777)
            outputs.chmod(0o777)
            (inputs / component.mutable_file).write_text(
                candidate.source_payload.replace("\r\n", "\n")
            )
            (inputs / "input.json").write_text(
                json.dumps(
                    configuration,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
            )
            command = self._base_command(inputs, outputs)
            command.extend(
                [
                    f"/input/{component.mutable_file}",
                    component.entry_point,
                    "/input/input.json",
                    "/output/output.json",
                    candidate.candidate_id,
                    payload_hash(self.policy),
                ]
            )
            try:
                completed = subprocess.run(
                    command,
                    capture_output=True,
                    timeout=policy.wall_time_seconds,
                    check=False,
                )
            except subprocess.TimeoutExpired:
                return CandidatePreparationResult(
                    candidate_id=candidate.candidate_id,
                    validation_status=CandidateValidationStatus.VALID,
                    execution_status=CandidateExecutionStatus.TIMED_OUT,
                    safe_error_code="candidate_timeout",
                    timeout=True,
                    runtime_seconds=round(time.monotonic() - started, 6),
                    cleanup_complete=True,
                )
            log = self._safe_log(
                completed.stdout + b"\n" + completed.stderr, policy.log_bytes
            )
            output_path = outputs / "output.json"
            if (
                completed.returncode != 0
                or not output_path.is_file()
                or output_path.stat().st_size > policy.output_bytes
            ):
                return CandidatePreparationResult(
                    candidate_id=candidate.candidate_id,
                    validation_status=CandidateValidationStatus.VALID,
                    execution_status=CandidateExecutionStatus.FAILED,
                    safe_error_code="candidate_execution_failed",
                    safe_log_excerpt=log,
                    runtime_seconds=round(time.monotonic() - started, 6),
                    cleanup_complete=True,
                )
            raw = output_path.read_bytes()
            envelope = json.loads(raw)
            if (
                not isinstance(envelope, dict)
                or set(envelope)
                != {"schema", "candidate_id", "execution_identity", "configuration"}
                or envelope["schema"] != "openevolve-candidate-output-v1"
                or envelope["candidate_id"] != candidate.candidate_id
                or envelope["execution_identity"] != payload_hash(self.policy)
                or not isinstance(envelope["configuration"], dict)
            ):
                raise ValueError("candidate_experiment_spec_invalid")
            output = envelope["configuration"]
            return CandidatePreparationResult(
                candidate_id=candidate.candidate_id,
                validation_status=CandidateValidationStatus.VALID,
                execution_status=CandidateExecutionStatus.COMPLETED,
                output_references=(f"executor-policy:{payload_hash(self.policy)}",),
                output_hashes=(hashlib.sha256(raw).hexdigest(),),
                generated_configuration=output,
                safe_log_excerpt=log,
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
            shutil.rmtree(root, ignore_errors=True)
