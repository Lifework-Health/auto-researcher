"""Transactional, identity-bound OpenEvolve search artefacts."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from auto_researcher.runtime.identity import payload_hash
from auto_researcher.search.openevolve.models import (
    OpenEvolveCandidate,
    OpenEvolvePopulationState,
    OpenEvolveSearchContract,
    OpenEvolveSearchResult,
)
from auto_researcher.tasks.artifacts import strict_json_bytes
from auto_researcher.tasks.models import TaskRuntimeContext

OPENEVOLVE_ARTEFACT_BUNDLE_VERSION = "openevolve-artefact-bundle-v1"
SEARCH_FILES = (
    "search_request.json",
    "search_manifest.json",
    "population_state.json",
    "candidate_index.json",
    "candidate_sources.json",
    "candidate_results.json",
    "lineage.json",
    "budget_summary.json",
    "stopping_summary.json",
)


@dataclass(frozen=True)
class OpenEvolveArtefactReceipt:
    references: tuple[str, ...]
    payload_hashes: dict[str, str]
    bundle_hash: str
    replayed: bool


def search_artefact_references(
    context: TaskRuntimeContext, search_request_id: str
) -> tuple[str, ...]:
    if context.output_dir is None or context.run_id is None:
        return ()
    prefix = Path("runs") / context.run_id / "openevolve" / search_request_id
    return tuple((prefix / name).as_posix() for name in SEARCH_FILES)


def _payloads(
    search_contract: OpenEvolveSearchContract,
    population: OpenEvolvePopulationState,
    result: OpenEvolveSearchResult,
    candidates: tuple[OpenEvolveCandidate, ...],
) -> tuple[dict[str, bytes], dict[str, str], str]:
    values = {
        "search_request.json": search_contract,
        "population_state.json": population,
        "candidate_index.json": {
            "protocol_version": "openevolve-candidate-index-v1",
            "candidates": [
                {
                    "candidate_id": item.candidate_id,
                    "source_hash": item.source_hash,
                    "status": item.status.value,
                    "generation": item.generation,
                    "parents": list(item.parent_candidate_ids),
                    "evaluation_identity": item.evaluation_identity,
                }
                for item in candidates
            ],
        },
        "candidate_sources.json": {
            "protocol_version": "openevolve-candidate-sources-v1",
            "representation": "structured-full-file-replacement-v1",
            "sources": [
                {
                    "candidate_id": item.candidate_id,
                    "mutable_file": item.mutable_file,
                    "source_hash": item.source_hash,
                    "source": item.source_payload,
                }
                for item in candidates
            ],
        },
        "candidate_results.json": {
            "protocol_version": "openevolve-candidate-results-v1",
            "results": [
                {
                    "candidate_id": item.candidate_id,
                    "validation": item.validation_result,
                    "preparation": item.preparation_result,
                    "evaluation_identity": item.evaluation_identity,
                }
                for item in candidates
            ],
        },
        "lineage.json": {
            "protocol_version": "openevolve-lineage-v1",
            "records": population.lineage,
        },
        "budget_summary.json": population.budget,
        "stopping_summary.json": result,
    }
    encoded = {name: strict_json_bytes(value) for name, value in values.items()}
    hashes = {
        name: hashlib.sha256(value).hexdigest() for name, value in encoded.items()
    }
    bundle_hash = payload_hash(
        {
            "schema": OPENEVOLVE_ARTEFACT_BUNDLE_VERSION,
            "payload_hashes": hashes,
        }
    )
    manifest = {
        "schema_version": OPENEVOLVE_ARTEFACT_BUNDLE_VERSION,
        "search_contract_hash": payload_hash(search_contract),
        "population_state_hash": payload_hash(population),
        "lineage_hash": payload_hash(population.lineage),
        "expected_filenames": list(SEARCH_FILES),
        "payload_hashes": hashes,
        "bundle_hash": bundle_hash,
        "completed": True,
    }
    return (
        {**encoded, "search_manifest.json": strict_json_bytes(manifest)},
        hashes,
        bundle_hash,
    )


def write_search_artefacts(
    context: TaskRuntimeContext,
    search_contract: OpenEvolveSearchContract,
    population: OpenEvolvePopulationState,
    result: OpenEvolveSearchResult,
    candidates: tuple[OpenEvolveCandidate, ...],
) -> OpenEvolveArtefactReceipt:
    references = search_artefact_references(context, search_contract.search_request_id)
    payloads, hashes, bundle_hash = _payloads(
        search_contract, population, result, candidates
    )
    if not references:
        return OpenEvolveArtefactReceipt((), hashes, bundle_hash, False)
    assert context.output_dir is not None and context.run_id is not None
    final = (
        context.output_dir
        / "runs"
        / context.run_id
        / "openevolve"
        / search_contract.search_request_id
    )
    final.parent.mkdir(parents=True, exist_ok=True)
    if final.exists():
        if {item.name for item in final.iterdir()} == set(SEARCH_FILES) and all(
            (final / name).read_bytes() == payload for name, payload in payloads.items()
        ):
            return OpenEvolveArtefactReceipt(references, hashes, bundle_hash, True)
        raise RuntimeError("conflicting_completed_openevolve_artefact_identity")
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{search_contract.search_request_id}.",
            suffix=".tmp",
            dir=final.parent,
        )
    )
    try:
        for name in SEARCH_FILES:
            destination = temporary / name
            with destination.open("xb") as handle:
                handle.write(payloads[name])
                handle.flush()
                os.fsync(handle.fileno())
        if {item.name for item in temporary.iterdir()} != set(SEARCH_FILES):
            raise RuntimeError("openevolve_artefact_bundle_incomplete")
        os.replace(temporary, final)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return OpenEvolveArtefactReceipt(references, hashes, bundle_hash, False)


def verify_search_artefacts(
    context: TaskRuntimeContext,
    search_request_id: str,
) -> tuple[bool, str | None]:
    if context.output_dir is None or context.run_id is None:
        return False, None
    directory = (
        context.output_dir / "runs" / context.run_id / "openevolve" / search_request_id
    )
    if not directory.is_dir() or {item.name for item in directory.iterdir()} != set(
        SEARCH_FILES
    ):
        return False, None
    try:
        manifest = json.loads(
            (directory / "search_manifest.json").read_text(encoding="utf-8")
        )
        if (
            manifest["schema_version"] != OPENEVOLVE_ARTEFACT_BUNDLE_VERSION
            or manifest["completed"] is not True
        ):
            return False, None
        hashes = {
            name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
            for name in SEARCH_FILES
            if name != "search_manifest.json"
        }
        bundle_hash = payload_hash(
            {"schema": OPENEVOLVE_ARTEFACT_BUNDLE_VERSION, "payload_hashes": hashes}
        )
        valid = (
            hashes == manifest["payload_hashes"]
            and bundle_hash == manifest["bundle_hash"]
        )
        return valid, bundle_hash if valid else None
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False, None
