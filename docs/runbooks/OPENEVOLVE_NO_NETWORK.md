# No-network verification

Run the real gated test, not a mock. From inside the candidate container it attempts DNS, outbound TCP to a documentation-only address, loopback and the metadata-service address. It also verifies controlled host sentinels, Docker socket, SSH material and credential variables are absent. Every network attempt must fail and every host asset must remain hidden. Failure returns a stable hardened-executor code and prevents candidate execution.
