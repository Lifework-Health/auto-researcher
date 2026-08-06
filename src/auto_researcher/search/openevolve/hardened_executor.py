"""Digest-bound Docker executor with a private inode-limited workspace."""

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

WORKER_PROTOCOL = "openevolve-hardened-worker-result-v2"
WORKSPACE_POLICY = "openevolve-workspace-policy-v1"


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def docker_policy(
    image_reference: str,
    image_digest: str,
    recipe: Path,
    worker: Path,
    runtime_version: str,
    candidate_child: Path | None = None,
) -> HardenedExecutorPolicy:
    child = candidate_child or worker.with_name("candidate_child.py")
    return HardenedExecutorPolicy(
        runtime_version=runtime_version,
        image_reference=image_reference,
        image_digest=image_digest,
        entrypoint_hash=file_hash(worker),
        candidate_child_hash=file_hash(child),
        build_recipe_hash=file_hash(recipe),
    )


class HardenedDockerExecutor:
    runner_id = "openevolve-hardened-executor-v2"

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
            inspection = subprocess.run(
                [
                    "docker",
                    "image",
                    "inspect",
                    self.policy.image_reference,
                    "--format",
                    "{{json .}}",
                ],
                capture_output=True,
                text=True,
                timeout=5,
                check=True,
            ).stdout.strip()
            image = json.loads(inspection)
        except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
            raise ValueError("hardened_executor_unavailable") from exc
        if version != self.policy.runtime_version:
            raise ValueError("hardened_executor_runtime_mismatch")
        if image.get("Id") != self.policy.image_digest:
            raise ValueError("hardened_executor_image_mismatch")
        config = image.get("Config", {})
        if (
            config.get("Entrypoint") != ["python", "/opt/runner/worker.py"]
            or config.get("User") != "65532:65532"
            or image.get("Architecture") not in {"arm64", "amd64"}
            or image.get("Os") != "linux"
        ):
            raise ValueError("hardened_executor_policy_violation")

    def _workspace_policy(
        self, candidate: OpenEvolveCandidate, policy: SandboxPolicy
    ) -> dict:
        return {
            "protocol_version": WORKSPACE_POLICY,
            "candidate_id": candidate.candidate_id,
            "executor_policy_identity": payload_hash(self.policy),
            "file_count_limit": policy.file_count_limit,
            "derived_inode_limit": policy.file_count_limit
            + self.policy.tmpfs_inode_overhead,
            "workspace_bytes": policy.workspace_bytes,
            "file_size_bytes": policy.file_size_bytes,
            "output_bytes": policy.output_bytes,
            "log_bytes": policy.log_bytes,
            "wall_time_seconds": policy.wall_time_seconds,
            "tmpfs": {
                "target": self.policy.workspace_mount,
                "rw": True,
                "noexec": True,
                "nosuid": True,
                "nodev": True,
                "mode": "0700",
                "uid": self.policy.fixed_uid,
                "gid": self.policy.fixed_gid,
            },
        }

    def _base_command(
        self,
        input_dir: Path,
        policy: SandboxPolicy,
        *,
        entrypoint: str | None = None,
    ) -> list[str]:
        inode_limit = policy.file_count_limit + self.policy.tmpfs_inode_overhead
        tmpfs = (
            f"{self.policy.workspace_mount}:rw,noexec,nosuid,nodev,mode=0700,"
            f"uid={self.policy.fixed_uid},gid={self.policy.fixed_gid},"
            f"size={policy.workspace_bytes},nr_inodes={inode_limit}"
        )
        command = [
            "docker",
            "run",
            "--rm",
            "--network",
            "none",
            "--read-only",
            "--user",
            f"{self.policy.fixed_uid}:{self.policy.fixed_gid}",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--pids-limit",
            str(policy.process_limit + self.policy.supervisor_pid_overhead),
            "--memory",
            str(policy.memory_bytes),
            "--cpus",
            "1",
            "--tmpfs",
            tmpfs,
            "--workdir",
            self.policy.workspace_mount,
            "--mount",
            f"type=bind,src={input_dir},dst={self.policy.input_mount},readonly",
            "--env",
            "HOME=/nonexistent",
            "--env",
            "TMPDIR=/workspace",
            "--env",
            "PYTHONDONTWRITEBYTECODE=1",
            "--env",
            "PYTHONNOUSERSITE=1",
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
    def _safe_log(output: bytes, limit: int) -> tuple[str, bool]:
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
        encoded = text.encode()
        return encoded[:limit].decode(errors="ignore"), len(encoded) > limit

    def verify_isolation(self) -> ExecutorIsolationResult:
        self._inspect()
        root = Path(
            tempfile.mkdtemp(prefix="openevolve-isolation-", dir=self.workspace_root)
        )
        try:
            inputs = root / "input"
            inputs.mkdir()
            policy = SandboxPolicy(
                policy_id=self.runner_id,
                cpu_time_seconds=2,
                wall_time_seconds=3,
                memory_bytes=256 * 1024 * 1024,
                process_limit=16,
                output_bytes=64_000,
                log_bytes=8_000,
                file_count_limit=8,
                workspace_bytes=1_048_576,
                file_size_bytes=64_000,
            )
            command = self._base_command(inputs, policy, entrypoint="python")
            command.append("/opt/runner/isolation_probe.py")
            completed = subprocess.run(
                command, capture_output=True, timeout=10, check=False
            )
            try:
                checks = (
                    json.loads(completed.stdout) if completed.returncode == 0 else {}
                )
            except json.JSONDecodeError:
                checks = {}
            inode_supported = checks.get("workspace_inode_limit") == 9
            workspace_bytes_supported = (
                checks.get("workspace_bytes_limit") == policy.workspace_bytes
            )
            network = all(
                checks.get(key) is True
                for key in (
                    "dns_denied",
                    "outbound_tcp_denied",
                    "outbound_ipv6_denied",
                    "http_denied",
                    "https_denied",
                    "loopback_denied",
                    "loopback_ipv6_denied",
                    "host_alias_denied",
                    "metadata_denied",
                    "raw_socket_denied",
                )
            )
            mounts = all(
                checks.get(key) is True
                for key in (
                    "host_sentinel_hidden",
                    "repository_hidden",
                    "docker_socket_hidden",
                    "containerd_socket_hidden",
                    "podman_socket_hidden",
                    "kubernetes_token_hidden",
                    "ssh_hidden",
                    "aws_hidden",
                    "config_hidden",
                    "tmp_read_only",
                    "var_tmp_read_only",
                    "home_unavailable",
                    "input_read_only",
                    "root_read_only",
                    "workspace_rw",
                    "workspace_noexec",
                    "workspace_nosuid",
                    "workspace_nodev",
                    "uid_non_root",
                    "gid_expected",
                    "supplementary_groups_expected",
                    "capabilities_absent",
                    "no_new_privileges",
                    "private_pid_namespace",
                    "cgroup_write_denied",
                )
            )
            environment = all(
                checks.get(key) is True
                for key in ("credential_names_absent", "host_pythonpath_absent")
            )
            code = None
            if not inode_supported:
                code = "hardened_executor_file_count_limit_unsupported"
            elif not workspace_bytes_supported:
                code = "hardened_executor_workspace_policy_mismatch"
            elif not (network and mounts and environment):
                code = "hardened_executor_network_isolation_unverified"
            return ExecutorIsolationResult(
                executor_policy_hash=payload_hash(self.policy),
                network_isolation_verified=network,
                mount_isolation_verified=(
                    mounts and inode_supported and workspace_bytes_supported
                ),
                environment_sanitisation_verified=environment,
                safe_checks=checks,
                safe_error_code=code,
            )
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def _failure(
        self,
        candidate: OpenEvolveCandidate,
        started: float,
        code: str,
        *,
        status: CandidateExecutionStatus = CandidateExecutionStatus.FAILED,
        timeout: bool = False,
        resource_limited: bool = False,
        evidence: dict | None = None,
    ) -> CandidatePreparationResult:
        evidence = evidence or {}
        return CandidatePreparationResult(
            protocol_version="candidate-preparation-v2",
            candidate_id=candidate.candidate_id,
            validation_status=CandidateValidationStatus.VALID,
            execution_status=status,
            safe_error_code=code,
            timeout=timeout,
            resource_limited=resource_limited,
            safe_log_excerpt=str(evidence.get("safe_log", "")),
            log_truncated=bool(evidence.get("log_truncated", False)),
            runtime_seconds=round(time.monotonic() - started, 6),
            cleanup_complete=True,
            executor_id=self.runner_id,
            executor_policy_identity=payload_hash(self.policy),
            execution_request_identity=evidence.get("execution_request_identity"),
            workspace_policy_identity=evidence.get("workspace_policy_identity"),
            worker_protocol_version=WORKER_PROTOCOL,
            supervisor_identity=self.policy.entrypoint_hash,
            image_digest=self.policy.image_digest,
            declared_file_count_limit=evidence.get("declared_file_count_limit"),
            derived_inode_limit=evidence.get("derived_inode_limit"),
            observed_workspace_entry_count=evidence.get(
                "observed_workspace_entry_count"
            ),
            observed_workspace_bytes=evidence.get("observed_workspace_bytes"),
            observed_max_file_bytes=evidence.get("observed_max_file_bytes"),
            workspace_bytes_limit=evidence.get("workspace_bytes_limit"),
            file_size_bytes_limit=evidence.get("file_size_bytes_limit"),
            observed_output_bytes=evidence.get("observed_output_bytes"),
            resource_limit_reason=code if resource_limited else None,
        )

    def accepts_preparation(
        self, preparation: CandidatePreparationResult, policy: SandboxPolicy
    ) -> bool:
        """Accept only evidence produced under this exact v2 execution boundary."""

        return (
            preparation.protocol_version == "candidate-preparation-v2"
            and preparation.executor_id == self.runner_id
            and preparation.executor_policy_identity == payload_hash(self.policy)
            and preparation.worker_protocol_version == WORKER_PROTOCOL
            and preparation.supervisor_identity == self.policy.entrypoint_hash
            and preparation.image_digest == self.policy.image_digest
            and preparation.declared_file_count_limit == policy.file_count_limit
            and preparation.derived_inode_limit
            == policy.file_count_limit + self.policy.tmpfs_inode_overhead
            and preparation.workspace_bytes_limit == policy.workspace_bytes
            and preparation.file_size_bytes_limit == policy.file_size_bytes
            and preparation.execution_request_identity is not None
            and preparation.workspace_policy_identity is not None
        )

    def prepare(
        self,
        candidate: OpenEvolveCandidate,
        component: EvolvableComponentSpec,
        policy: SandboxPolicy,
        configuration: dict,
    ) -> CandidatePreparationResult:
        started = time.monotonic()
        if candidate.preparation_result is not None:
            if self.accepts_preparation(candidate.preparation_result, policy):
                return candidate.preparation_result
            return self._failure(
                candidate,
                started,
                "candidate_preparation_identity_mismatch",
            )
        workspace_policy = self._workspace_policy(candidate, policy)
        workspace_identity = payload_hash(workspace_policy)
        execution_request = {
            "protocol_version": "openevolve-execution-request-v2",
            "candidate_id": candidate.candidate_id,
            "source_hash": candidate.source_hash,
            "component_interface_hash": candidate.component_interface_hash,
            "configuration_hash": payload_hash(configuration),
            "executor_policy_identity": payload_hash(self.policy),
            "workspace_policy_identity": workspace_identity,
            "image_digest": self.policy.image_digest,
        }
        request_identity = payload_hash(execution_request)
        base_evidence = {
            "execution_request_identity": request_identity,
            "workspace_policy_identity": workspace_identity,
            "declared_file_count_limit": policy.file_count_limit,
            "derived_inode_limit": policy.file_count_limit
            + self.policy.tmpfs_inode_overhead,
            "workspace_bytes_limit": policy.workspace_bytes,
            "file_size_bytes_limit": policy.file_size_bytes,
        }
        try:
            isolation = self.verify_isolation()
            if isolation.safe_error_code:
                raise ValueError(isolation.safe_error_code)
        except ValueError as exc:
            return self._failure(candidate, started, str(exc), evidence=base_evidence)
        root = Path(
            tempfile.mkdtemp(prefix="openevolve-hardened-", dir=self.workspace_root)
        )
        try:
            inputs = root / "input"
            inputs.mkdir()
            (inputs / component.mutable_file).write_text(
                candidate.source_payload.replace("\r\n", "\n"),
                encoding="utf-8",
                newline="\n",
            )
            (inputs / "input.json").write_text(
                json.dumps(
                    configuration,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ),
                encoding="utf-8",
            )
            command = self._base_command(inputs, policy)
            command.extend(
                [
                    f"/input/{component.mutable_file}",
                    component.entry_point,
                    "/input/input.json",
                    candidate.candidate_id,
                    request_identity,
                    payload_hash(self.policy),
                    self.policy.image_digest,
                    str(policy.file_count_limit),
                    str(policy.file_count_limit + self.policy.tmpfs_inode_overhead),
                    str(policy.workspace_bytes),
                    str(policy.output_bytes),
                    str(policy.file_size_bytes),
                    str(policy.log_bytes),
                    str(policy.cpu_time_seconds),
                ]
            )
            try:
                completed = subprocess.run(
                    command,
                    capture_output=True,
                    timeout=policy.wall_time_seconds,
                    check=False,
                )
            except subprocess.TimeoutExpired as exc:
                log, truncated = self._safe_log(
                    (exc.stdout or b"") + b"\n" + (exc.stderr or b""),
                    policy.log_bytes,
                )
                return self._failure(
                    candidate,
                    started,
                    "candidate_timeout",
                    status=CandidateExecutionStatus.TIMED_OUT,
                    timeout=True,
                    evidence={
                        **base_evidence,
                        "safe_log": log,
                        "log_truncated": truncated,
                    },
                )
            if completed.returncode == 137 and not completed.stdout:
                return self._failure(
                    candidate,
                    started,
                    "candidate_memory_limit",
                    status=CandidateExecutionStatus.RESOURCE_LIMITED,
                    resource_limited=True,
                    evidence=base_evidence,
                )
            if len(completed.stdout) > 1_000_000:
                return self._failure(
                    candidate,
                    started,
                    "hardened_executor_worker_protocol_invalid",
                    evidence=base_evidence,
                )
            lines = completed.stdout.splitlines()
            if len(lines) != 1 or completed.stderr:
                return self._failure(
                    candidate,
                    started,
                    "hardened_executor_worker_protocol_invalid",
                    evidence=base_evidence,
                )
            try:
                envelope = json.loads(lines[0])
            except json.JSONDecodeError:
                return self._failure(
                    candidate,
                    started,
                    "hardened_executor_worker_protocol_invalid",
                    evidence=base_evidence,
                )
            expected = {
                "protocol_version": WORKER_PROTOCOL,
                "candidate_id": candidate.candidate_id,
                "execution_request_identity": request_identity,
                "executor_policy_identity": payload_hash(self.policy),
                "image_identity": self.policy.image_digest,
                "declared_file_count_limit": policy.file_count_limit,
                "derived_inode_limit": policy.file_count_limit
                + self.policy.tmpfs_inode_overhead,
                "workspace_bytes_limit": policy.workspace_bytes,
            }
            if not isinstance(envelope, dict) or any(
                envelope.get(key) != value for key, value in expected.items()
            ):
                return self._failure(
                    candidate,
                    started,
                    "hardened_executor_worker_protocol_invalid",
                    evidence=base_evidence,
                )
            evidence = {
                **base_evidence,
                "safe_log": envelope.get("safe_log", ""),
                "log_truncated": envelope.get("log_truncated", False),
                "observed_workspace_entry_count": envelope.get(
                    "observed_workspace_entry_count"
                ),
                "observed_workspace_bytes": envelope.get("observed_workspace_bytes"),
                "observed_max_file_bytes": envelope.get("observed_max_file_bytes"),
                "observed_output_bytes": envelope.get("observed_output_bytes"),
            }
            if envelope.get("status") != "COMPLETED":
                code = str(
                    envelope.get(
                        "safe_error_code", "hardened_executor_worker_protocol_invalid"
                    )
                )
                limited = bool(envelope.get("resource_limited"))
                return self._failure(
                    candidate,
                    started,
                    code,
                    status=(
                        CandidateExecutionStatus.RESOURCE_LIMITED
                        if limited
                        else CandidateExecutionStatus.FAILED
                    ),
                    resource_limited=limited,
                    evidence=evidence,
                )
            configuration_result = envelope.get("configuration")
            if not isinstance(configuration_result, dict):
                return self._failure(
                    candidate,
                    started,
                    "hardened_executor_worker_protocol_invalid",
                    evidence=evidence,
                )
            canonical = json.dumps(
                configuration_result,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode()
            if (
                envelope.get("configuration_hash")
                != hashlib.sha256(canonical).hexdigest()
                or envelope.get("observed_output_bytes") != len(canonical)
                or len(canonical) > policy.output_bytes
            ):
                return self._failure(
                    candidate,
                    started,
                    "hardened_executor_worker_protocol_invalid",
                    evidence=evidence,
                )
            return CandidatePreparationResult(
                protocol_version="candidate-preparation-v2",
                candidate_id=candidate.candidate_id,
                validation_status=CandidateValidationStatus.VALID,
                execution_status=CandidateExecutionStatus.COMPLETED,
                output_references=(f"executor-policy:{payload_hash(self.policy)}",),
                output_hashes=(hashlib.sha256(lines[0]).hexdigest(),),
                generated_configuration=configuration_result,
                safe_log_excerpt=str(envelope.get("safe_log", "")),
                log_truncated=bool(envelope.get("log_truncated", False)),
                runtime_seconds=round(time.monotonic() - started, 6),
                cleanup_complete=True,
                executor_id=self.runner_id,
                executor_policy_identity=payload_hash(self.policy),
                execution_request_identity=request_identity,
                workspace_policy_identity=workspace_identity,
                worker_protocol_version=WORKER_PROTOCOL,
                supervisor_identity=self.policy.entrypoint_hash,
                image_digest=self.policy.image_digest,
                declared_file_count_limit=policy.file_count_limit,
                derived_inode_limit=policy.file_count_limit
                + self.policy.tmpfs_inode_overhead,
                observed_workspace_entry_count=envelope.get(
                    "observed_workspace_entry_count"
                ),
                observed_workspace_bytes=envelope.get("observed_workspace_bytes"),
                observed_max_file_bytes=envelope.get("observed_max_file_bytes"),
                workspace_bytes_limit=policy.workspace_bytes,
                file_size_bytes_limit=policy.file_size_bytes,
                observed_output_bytes=len(canonical),
            )
        except (OSError, ValueError, json.JSONDecodeError):
            return self._failure(
                candidate,
                started,
                "candidate_execution_failed",
                evidence=base_evidence,
            )
        finally:
            shutil.rmtree(root, ignore_errors=True)
