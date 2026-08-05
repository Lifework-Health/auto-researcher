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


result = {
    "schema": "executor-isolation-result-v1",
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
}
Path("/output/isolation.json").write_text(
    json.dumps(result, sort_keys=True, separators=(",", ":"))
)
