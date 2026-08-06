FROM python@sha256:519591d6871b7bc437060736b9f7456b8731f1499a57e22e6c285135ae657bf7

LABEL org.opencontainers.image.title="Auto Researcher OpenEvolve candidate executor"
LABEL org.opencontainers.image.version="openevolve-hardened-executor-v2"

COPY worker.py /opt/runner/worker.py
COPY candidate_child.py /opt/runner/candidate_child.py
COPY isolation_probe.py /opt/runner/isolation_probe.py
RUN chmod 0444 /opt/runner/worker.py /opt/runner/candidate_child.py /opt/runner/isolation_probe.py

USER 65532:65532
WORKDIR /workspace
ENTRYPOINT ["python", "/opt/runner/worker.py"]
