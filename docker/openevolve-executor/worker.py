"""Image-owned supervisor for the hardened OpenEvolve executor v2."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import subprocess
import sys
import threading
from pathlib import Path

PROTOCOL = "openevolve-hardened-worker-result-v2"
CHILD = "/opt/runner/candidate_child.py"
WORKSPACE = Path("/workspace")


def emit(payload: dict, exit_code: int) -> None:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    if len(encoded) > 1_000_000:
        payload = {
            "protocol_version": PROTOCOL,
            "candidate_id": str(payload.get("candidate_id", "unknown"))[:256],
            "status": "ERROR",
            "safe_error_code": "hardened_executor_worker_protocol_invalid",
            "resource_limited": False,
            "log_truncated": True,
            "safe_log": "",
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    sys.stdout.buffer.write(encoded + b"\n")
    sys.stdout.buffer.flush()
    raise SystemExit(exit_code)


def safe_log(stdout: bytes, stderr: bytes, limit: int) -> tuple[str, bool]:
    raw = (stdout + b"\n" + stderr).decode("utf-8", errors="replace")
    raw = re.sub(
        r"Traceback \(most recent call last\):.*",
        "candidate_execution_failed",
        raw,
        flags=re.S,
    )
    raw = re.sub(r"/(?:Users|home|private|tmp|var)/[^\s:]+", "<redacted-path>", raw)
    encoded = raw.encode()
    return encoded[:limit].decode(errors="ignore").strip(), len(encoded) > limit


def read_capped(stream, limit: int, state: dict, key: str) -> None:
    kept = bytearray()
    exceeded = False
    while True:
        chunk = stream.read(8192)
        if not chunk:
            break
        remaining = max(0, limit - len(kept))
        kept.extend(chunk[:remaining])
        exceeded = exceeded or len(chunk) > remaining
    state[key] = bytes(kept)
    state[f"{key}_truncated"] = exceeded


def workspace_entries() -> tuple[int, bool]:
    count = 0
    valid = True
    for root, directories, files in os.walk(WORKSPACE, topdown=True, followlinks=False):
        for name in [*directories, *files]:
            count += 1
            mode = os.lstat(Path(root, name)).st_mode
            if not (stat.S_ISREG(mode) or stat.S_ISDIR(mode)):
                valid = False
    return count, valid


def base_payload(candidate_id: str) -> dict:
    return {
        "protocol_version": PROTOCOL,
        "candidate_id": candidate_id,
        "resource_limited": False,
        "log_truncated": False,
        "safe_log": "",
    }


def main() -> None:
    if len(sys.argv) != 15:
        emit(
            {
                **base_payload("unknown"),
                "status": "ERROR",
                "safe_error_code": "hardened_executor_worker_protocol_invalid",
            },
            64,
        )
    (
        source,
        entry_point,
        input_path,
        candidate_id,
        execution_request_identity,
        executor_policy_identity,
        image_identity,
        file_count_text,
        inode_limit_text,
        workspace_bytes_text,
        output_bytes_text,
        file_size_bytes_text,
        log_bytes_text,
        cpu_time_text,
    ) = sys.argv[1:]
    try:
        file_count_limit = int(file_count_text)
        inode_limit = int(inode_limit_text)
        workspace_bytes = int(workspace_bytes_text)
        output_bytes = int(output_bytes_text)
        file_size_bytes = int(file_size_bytes_text)
        log_bytes = int(log_bytes_text)
        cpu_time_seconds = int(cpu_time_text)
        if (
            file_count_limit <= 0
            or inode_limit != file_count_limit + 1
            or workspace_bytes <= 0
            or output_bytes <= 0
            or file_size_bytes <= 0
            or log_bytes <= 0
            or cpu_time_seconds <= 0
            or not source.startswith("/input/")
            or input_path != "/input/input.json"
            or not entry_point.isidentifier()
        ):
            raise ValueError
        filesystem = os.statvfs(WORKSPACE)
        if filesystem.f_files != inode_limit:
            emit(
                {
                    **base_payload(candidate_id),
                    "status": "ERROR",
                    "safe_error_code": "hardened_executor_file_count_limit_unsupported",
                    "declared_file_count_limit": file_count_limit,
                    "derived_inode_limit": inode_limit,
                },
                78,
            )
        if filesystem.f_blocks * filesystem.f_frsize != workspace_bytes:
            emit(
                {
                    **base_payload(candidate_id),
                    "status": "ERROR",
                    "safe_error_code": "hardened_executor_workspace_policy_mismatch",
                    "workspace_bytes_limit": workspace_bytes,
                },
                78,
            )
    except (OSError, ValueError):
        emit(
            {
                **base_payload(candidate_id),
                "status": "ERROR",
                "safe_error_code": "hardened_executor_workspace_policy_mismatch",
            },
            64,
        )

    read_fd, write_fd = os.pipe()
    command = [
        sys.executable,
        "-I",
        "-S",
        CHILD,
        source,
        entry_point,
        input_path,
        str(write_fd),
        str(output_bytes),
        str(file_size_bytes),
        str(cpu_time_seconds),
    ]
    child = subprocess.Popen(
        command,
        cwd=WORKSPACE,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        close_fds=True,
        pass_fds=(write_fd,),
        env={
            "HOME": "/nonexistent",
            "TMPDIR": "/workspace",
            "TZ": "UTC",
            "LC_ALL": "C",
            "LANG": "C",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
        },
    )
    os.close(write_fd)
    control = os.fdopen(read_fd, "rb", buffering=0)
    captures: dict[str, bytes | bool] = {}
    threads = [
        threading.Thread(
            target=read_capped, args=(child.stdout, log_bytes, captures, "stdout")
        ),
        threading.Thread(
            target=read_capped, args=(child.stderr, log_bytes, captures, "stderr")
        ),
        threading.Thread(
            target=read_capped, args=(control, output_bytes, captures, "control")
        ),
    ]
    for thread in threads:
        thread.start()
    return_code = child.wait()
    for thread in threads:
        thread.join()
    count, file_types_valid = workspace_entries()
    maximum_file_size = max(
        (path.stat().st_size for path in WORKSPACE.rglob("*") if path.is_file()),
        default=0,
    )
    total_file_size = sum(
        path.stat().st_size for path in WORKSPACE.rglob("*") if path.is_file()
    )
    log, log_truncated = safe_log(
        captures.get("stdout", b""), captures.get("stderr", b""), log_bytes
    )
    log_truncated = (
        log_truncated
        or bool(captures.get("stdout_truncated"))
        or bool(captures.get("stderr_truncated"))
    )
    common = {
        **base_payload(candidate_id),
        "execution_request_identity": execution_request_identity,
        "executor_policy_identity": executor_policy_identity,
        "image_identity": image_identity,
        "declared_file_count_limit": file_count_limit,
        "derived_inode_limit": inode_limit,
        "observed_workspace_entry_count": count,
        "observed_workspace_bytes": total_file_size,
        "observed_max_file_bytes": maximum_file_size,
        "workspace_bytes_limit": workspace_bytes,
        "safe_log": log,
        "log_truncated": log_truncated,
    }
    if not file_types_valid:
        emit(
            {
                **common,
                "status": "ERROR",
                "safe_error_code": "candidate_file_type_forbidden",
            },
            65,
        )
    if count > file_count_limit:
        emit(
            {
                **common,
                "status": "RESOURCE_LIMITED",
                "safe_error_code": "candidate_file_count_limit",
                "resource_limited": True,
            },
            75,
        )
    if maximum_file_size > file_size_bytes:
        emit(
            {
                **common,
                "status": "RESOURCE_LIMITED",
                "safe_error_code": "candidate_file_size_limit",
                "resource_limited": True,
            },
            75,
        )
    if total_file_size > workspace_bytes:
        emit(
            {
                **common,
                "status": "RESOURCE_LIMITED",
                "safe_error_code": "candidate_workspace_size_limit",
                "resource_limited": True,
            },
            75,
        )
    if return_code != 0:
        filesystem = os.statvfs(WORKSPACE)
        pids_events = Path("/sys/fs/cgroup/pids.events")
        pids_limited = pids_events.is_file() and any(
            line.startswith("max ") and int(line.split()[1]) > 0
            for line in pids_events.read_text().splitlines()
        )
        memory_events = Path("/sys/fs/cgroup/memory.events")
        memory_limited = memory_events.is_file() and any(
            line.startswith(("oom ", "oom_kill ")) and int(line.split()[1]) > 0
            for line in memory_events.read_text().splitlines()
        )
        if count >= file_count_limit and filesystem.f_favail == 0:
            code = "candidate_file_count_limit"
        elif maximum_file_size >= file_size_bytes:
            code = "candidate_file_size_limit"
        elif filesystem.f_bavail * filesystem.f_frsize < 4096:
            code = "candidate_workspace_size_limit"
        elif pids_limited:
            code = "candidate_process_limit"
        elif memory_limited:
            code = "candidate_memory_limit"
        elif return_code < 0:
            code = "candidate_cpu_limit"
        else:
            code = "candidate_execution_failed"
        emit(
            {
                **common,
                "status": "RESOURCE_LIMITED"
                if code != "candidate_execution_failed"
                else "ERROR",
                "safe_error_code": code,
                "resource_limited": code != "candidate_execution_failed",
            },
            75 if code != "candidate_execution_failed" else 70,
        )
    if captures.get("control_truncated"):
        emit(
            {
                **common,
                "status": "RESOURCE_LIMITED",
                "safe_error_code": "candidate_output_limit",
                "resource_limited": True,
            },
            75,
        )
    try:
        configuration = json.loads(captures.get("control", b""))
        if not isinstance(configuration, dict):
            raise ValueError
        canonical = json.dumps(
            configuration, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
        if len(canonical) > output_bytes:
            raise OverflowError
    except OverflowError:
        emit(
            {
                **common,
                "status": "RESOURCE_LIMITED",
                "safe_error_code": "candidate_output_limit",
                "resource_limited": True,
            },
            75,
        )
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError):
        emit(
            {
                **common,
                "status": "ERROR",
                "safe_error_code": "candidate_experiment_spec_invalid",
            },
            65,
        )
    emit(
        {
            **common,
            "status": "COMPLETED",
            "configuration": configuration,
            "configuration_hash": hashlib.sha256(canonical).hexdigest(),
            "observed_output_bytes": len(canonical),
        },
        0,
    )


if __name__ == "__main__":
    main()
