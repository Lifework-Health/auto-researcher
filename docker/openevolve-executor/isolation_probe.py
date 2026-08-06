"""Independent image-owned isolation probe; emits names and booleans only."""

import json
import os
import socket
import urllib.request
from pathlib import Path


def denied(operation):
    try:
        operation()
    except Exception:
        return True
    return False


def connection_denied(address, family=socket.AF_INET):
    def connect():
        handle = socket.socket(family, socket.SOCK_STREAM)
        handle.settimeout(0.2)
        try:
            handle.connect(address)
        finally:
            handle.close()

    return denied(connect)


status = {}
for line in Path("/proc/self/status").read_text().splitlines():
    if ":" in line:
        key, value = line.split(":", 1)
        status[key] = value.strip()
mount_lines = Path("/proc/mounts").read_text().splitlines()
root_mount = next(line for line in mount_lines if line.split()[1] == "/")
workspace_mount = next(line for line in mount_lines if line.split()[1] == "/workspace")
root_options = set(root_mount.split()[3].split(","))
workspace_options = set(workspace_mount.split()[3].split(","))
workspace = os.statvfs("/workspace")
environment_names = sorted(os.environ)
forbidden_environment_markers = (
    "ANTHROPIC",
    "OPENAI",
    "NEO4J",
    "AWS",
    "GOOGLE",
    "GCP",
    "AZURE",
    "GITHUB",
    "SSH",
    "DOCKER",
    "PROXY",
    "TOKEN",
    "SECRET",
    "PASSWORD",
    "API_KEY",
)

result = {
    "schema": "executor-isolation-result-v2",
    "dns_denied": denied(lambda: socket.getaddrinfo("example.com", 443)),
    "outbound_tcp_denied": connection_denied(("192.0.2.1", 9)),
    "outbound_ipv6_denied": connection_denied(
        ("2001:db8::1", 9, 0, 0), socket.AF_INET6
    ),
    "http_denied": denied(
        lambda: urllib.request.urlopen("http://example.com", timeout=0.2)
    ),
    "https_denied": denied(
        lambda: urllib.request.urlopen("https://example.com", timeout=0.2)
    ),
    "loopback_denied": connection_denied(("127.0.0.1", 9)),
    "loopback_ipv6_denied": connection_denied(("::1", 9, 0, 0), socket.AF_INET6),
    "host_alias_denied": denied(lambda: socket.getaddrinfo("host.docker.internal", 80)),
    "metadata_denied": connection_denied(("169.254.169.254", 80)),
    "raw_socket_denied": denied(
        lambda: socket.socket(socket.AF_INET, socket.SOCK_RAW, socket.IPPROTO_ICMP)
    ),
    "host_sentinel_hidden": not Path("/host-sentinel/secret.txt").exists(),
    "repository_hidden": not any(
        Path(path).exists() for path in ("/repo", "/source", "/app")
    ),
    "docker_socket_hidden": not Path("/var/run/docker.sock").exists(),
    "containerd_socket_hidden": not Path("/run/containerd/containerd.sock").exists(),
    "podman_socket_hidden": not Path("/run/podman/podman.sock").exists(),
    "kubernetes_token_hidden": not Path(
        "/var/run/secrets/kubernetes.io/serviceaccount/token"
    ).exists(),
    "ssh_hidden": not Path.home().joinpath(".ssh").exists(),
    "aws_hidden": not Path.home().joinpath(".aws").exists(),
    "config_hidden": not Path.home().joinpath(".config").exists(),
    "credential_names_absent": not any(
        any(marker in name.upper() for marker in forbidden_environment_markers)
        for name in environment_names
    ),
    "environment_names": environment_names,
    "host_pythonpath_absent": "PYTHONPATH" not in environment_names,
    "tmp_read_only": denied(lambda: Path("/tmp/escape").write_text("x")),
    "var_tmp_read_only": denied(lambda: Path("/var/tmp/escape").write_text("x")),
    "home_unavailable": not Path.home().exists(),
    "input_read_only": denied(lambda: Path("/input/escape").write_text("x")),
    "root_read_only": "ro" in root_options,
    "workspace_rw": "rw" in workspace_options,
    "workspace_noexec": "noexec" in workspace_options,
    "workspace_nosuid": "nosuid" in workspace_options,
    "workspace_nodev": "nodev" in workspace_options,
    "workspace_inode_limit": workspace.f_files,
    "workspace_free_inodes": workspace.f_favail,
    "workspace_bytes_limit": workspace.f_blocks * workspace.f_frsize,
    "workspace_uid": os.stat("/workspace").st_uid,
    "workspace_gid": os.stat("/workspace").st_gid,
    "uid_non_root": os.getuid() == 65532 and os.geteuid() == 65532,
    "gid_expected": os.getgid() == 65532 and os.getegid() == 65532,
    "supplementary_groups_expected": os.getgroups() == [65532],
    "capabilities_absent": status.get("CapEff") == "0000000000000000",
    "no_new_privileges": status.get("NoNewPrivs") == "1",
    "private_pid_namespace": Path("/proc/1/cmdline").read_bytes().startswith(b"python"),
    "cgroup_write_denied": denied(
        lambda: Path("/sys/fs/cgroup/cgroup.procs").write_text("1")
    ),
}
print(json.dumps(result, sort_keys=True, separators=(",", ":")))
