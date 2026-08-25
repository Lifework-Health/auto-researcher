"""Load and render repository-versioned prompt templates."""

from __future__ import annotations

import hashlib
from importlib import resources

from pydantic import BaseModel, ConfigDict


class PromptBundle(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    version: str
    system_template: str
    user_template: str
    system_hash: str
    user_hash: str
    prompt_hash: str

    def render(self, *, context_json: str, correction: str = "") -> tuple[str, str]:
        user = self.user_template.replace("{{context_json}}", context_json).replace(
            "{{correction}}", correction
        )
        return self.system_template, user


def load_prompt(name: str, version: str = "1.0.0") -> PromptBundle:
    versions = {"1.0.0": "v1", "2.0.0": "v2"}
    if (
        name not in {"hypothesis", "planner", "research_director"}
        or version not in versions
    ):
        raise ValueError(f"unknown prompt {name}@{version}")
    root = resources.files("auto_researcher.prompts").joinpath(name)
    stem = versions[version]
    system = root.joinpath(f"{stem}_system.md").read_text(encoding="utf-8")
    user = root.joinpath(f"{stem}_user.md").read_text(encoding="utf-8")
    system_hash = hashlib.sha256(system.encode()).hexdigest()
    user_hash = hashlib.sha256(user.encode()).hexdigest()
    prompt_hash = hashlib.sha256(f"{system_hash}:{user_hash}".encode()).hexdigest()
    return PromptBundle(
        name=name,
        version=version,
        system_template=system,
        user_template=user,
        system_hash=system_hash,
        user_hash=user_hash,
        prompt_hash=prompt_hash,
    )
