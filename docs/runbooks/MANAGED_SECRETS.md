# Managed secrets

Auto Researcher separates a secret's non-sensitive identity from its resolved
runtime value. Task configuration may contain a `SecretReference`; it must never
contain the value. Resolution happens only while constructing the runtime
provider client. Resolved values are redacted in `str`/`repr`, cannot be pickled,
and are not fields on research contracts, model-call configuration, prompts,
approval artefacts, checkpoints, or provenance events.

## Production pattern on Compute Engine

The recommended production path is:

```text
Compute Engine attached service account
  -> Secret Manager accessor permission on one required secret
  -> Auto Researcher GoogleSecretManagerProvider using ADC
  -> runtime Anthropic provider credential
```

For the current worker, the project is `auto-researcherv22` and the attached
identity is
`auto-researcher-worker@auto-researcherv22.iam.gserviceaccount.com`. An operator
or project administrator must enable the Secret Manager API and grant
`roles/secretmanager.secretAccessor`. Scope that role to the specific secret, or
as narrowly as practical. Auto Researcher does not enable APIs, grant IAM roles,
or weaken access policy.

Compute Engine uses Application Default Credentials from the attached service
account. Service-account JSON key files are not required and must not be placed
in the repository. Install the optional client only on workers that use it:

```bash
pip install -e '.[secrets-gcp]'
```

Configure the live agent with identity only:

```yaml
agents:
  mode: live
  provider: anthropic
  credential:
    logical_name: anthropic_api_key
    provider: google_secret_manager
    provider_identifier: projects/auto-researcherv22/secrets/anthropic-api-key
    version: latest  # or an explicit numeric version such as "7"
    required: true
```

Google Secret Manager references in standard configuration must use the full
`projects/<project>/secrets/<secret>` identifier. Bare secret names and implicit
project selection are rejected. The provider appends `/versions/latest` when
`version` is omitted, or the configured version when it is present.

Missing secrets, authentication failure, permission denial, disabled API,
not-found versions, timeouts, provider unavailability, and malformed, empty, or
non-UTF-8 payloads fail closed with a bounded application error. Raw Google
exceptions are not chained or printed.
Secret Manager and its Google authentication packages remain optional for core
and offline installations.

Changing the value behind an environment variable or Secret Manager version is
an operational rotation. The value is never hashed into the research contract,
initial graph input, experiment configuration, or scientific identity. A
configured provider/name/version reference can be retained as non-sensitive
operational audit metadata, but it is not scientific evidence.

The credential is resolved once while assembling a live runtime and that same
runtime-only value is supplied to both the hypothesis and planner clients. A
new runtime assembly resolves again, so a restart or fresh assembly observes a
rotated environment value or managed-secret version. The runtime does not poll
or refresh a credential inside an existing assembly.

## Environment and local fallback

Omitting `agents.credential` preserves the existing `ANTHROPIC_API_KEY`
behaviour. An explicit equivalent is:

```yaml
credential:
  logical_name: anthropic_api_key
  provider: environment
  provider_identifier: ANTHROPIC_API_KEY
  required: true
```

Explicit environment references must always include `provider_identifier`;
there is no fallback from `logical_name` to an environment variable name.

For development or operator recovery, a protected file may bootstrap the shell
environment. Auto Researcher intentionally does not parse shell files:

```bash
mkdir -p ~/.config/auto-researcher
touch ~/.config/auto-researcher/secrets.env
chmod 0600 ~/.config/auto-researcher/secrets.env
```

Keep the file outside the repository, source it from the user's shell startup,
and add `secrets.env`/`*.secrets.env` to both repository and global ignore rules.
Verify permissions after editing or copying it. This local `0600` environment
file is a development/operator fallback, not the target production mechanism.
Never commit secrets, put them in task YAML, paste them into CLI diagnostics, or
record them in logs, prompts, checkpoints, approvals, provenance, or artefacts.

The protected file should contain an exported variable, for example
`export ANTHROPIC_API_KEY=...`; source the file by path from `.bashrc` without
printing it. Keep the value out of shell history while creating or rotating the
file.

Future credential consumers can reuse `SecretReference` and the narrow
`SecretProvider.resolve` protocol. Cross-provider lease renewal, dynamic
credential issuance, automatic IAM provisioning, and a general vault framework
are intentionally outside this subsystem.
