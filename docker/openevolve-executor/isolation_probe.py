"""Independent image-owned isolation probe; emits safe JSON to stdout."""

import json
import os
import socket
from pathlib import Path


def denied(operation):
    try:
        operation()
    except Exception:
        return True
    return False


workspace = os.statvfs("/workspace")
result = {
    "schema": "executor-isolation-result-v2",
    "dns_denied": denied(lambda: socket.getaddrinfo("example.invalid", 443)),
    "outbound_tcp_denied": denied(
        lambda: socket.create_connection(("192.0.2.1", 9), 0.2)
    ),
    "loopback_denied": denied(lambda: socket.create_connection(("127.0.0.1", 9), 0.2)),
    "metadata_denied": denied(
        lambda: socket.create_connection(("169.254.169.254", 80), 0.2)
    ),
    "host_sentinel_hidden": not Path("/host-sentinel/secret.txt").exists(),
    "docker_socket_hidden": not Path("/var/run/docker.sock").exists(),
    "ssh_hidden": not Path.home().joinpath(".ssh").exists(),
    "credential_names_absent": not any(
        os.environ.get(key)
        for key in (
            "ANTHROPIC_API_KEY",
            "OPENAI_API_KEY",
            "NEO4J_PASSWORD",
            "AWS_SECRET_ACCESS_KEY",
            "GOOGLE_APPLICATION_CREDENTIALS",
        )
    ),
    "tmp_read_only": denied(lambda: Path("/tmp/escape").write_text("x")),
    "var_tmp_read_only": denied(lambda: Path("/var/tmp/escape").write_text("x")),
    "home_unavailable": not Path.home().exists(),
    "workspace_inode_limit": workspace.f_files,
    "workspace_free_inodes": workspace.f_favail,
    "workspace_bytes_limit": workspace.f_blocks * workspace.f_frsize,
    "workspace_uid": os.stat("/workspace").st_uid,
    "workspace_gid": os.stat("/workspace").st_gid,
}
print(json.dumps(result, sort_keys=True, separators=(",", ":")))
