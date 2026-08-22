"""Controller-owned search portfolio and staged-fidelity graduation policy."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any

from auto_researcher.contracts.enums import (
    EvidenceStatus,
    EventType,
    ProposalSource,
    SearchType,
)
from auto_researcher.contracts.models import DecisionEvent, SearchRequest
from auto_researcher.runtime.identity import payload_hash
from auto_researcher.tasks.feta_unet_search.configuration import (
    AUGMENTATION_POLICIES,
    CANDIDATE_CONFIGURATION_FIELDS,
    DICE_WEIGHT_BOUNDS,
    DROPOUT_BOUNDS,
    LEARNING_RATE_BOUNDS,
    LOSS_VARIANTS,
    MODEL_VARIANTS,
    V6_ARCHITECTURE_BUDGET,
    V6_BASIC_UNET_FEATURE_PROFILES,
    V6_OPTUNA_FEATURE_PROFILES,
    V7_ARCHITECTURE_BUDGET,
    WEIGHT_DECAY_BOUNDS,
    FeTAUNetSearchConfiguration,
)
from auto_researcher.tasks.feta_unet_search.continuation import trajectory_identity
from auto_researcher.tasks.feta_unet_search.openevolve import (
    default_openevolve_configuration,
    policy_from_configuration,
)
from auto_researcher.tasks.models import TaskRuntimeContext

PORTFOLIO_VERSION = "feta-unet-60-18-7-2-portfolio-v1"
TREE_PORTFOLIO_VERSION = "feta-unet-family-lineage-tree-24-18-6-8-4-2-v3"
V6_TREE_PORTFOLIO_VERSION = "feta-basicunet-architecture-tree-12-18-6-10-5-3-v1"
V7_MECHANISM_PORTFOLIO_VERSION = "feta-basicunet-structural-tree-4-12-8-2-8-4-2-v2"
V7_REQ11_DIAGNOSTIC_SCHEMA_VERSION = "feta-unet-diagnostic-report-v1"
V7_REQ11_PANEL_IDENTITY = (
    "c2d6839bd16b292322fe97bbc71cb4f0333305b5a379bd2c8e4d3544e232b871"
)
V7_REQ11_BASELINE_EXPERIMENT_ID = "experiment-fc2d8d2a371ddba0"
V7_REQ11_CANDIDATE_EXPERIMENT_IDS = (
    "experiment-643b2c5f65b4bc25",
    "experiment-ccc5fcf318cf2eb1",
)
V7_REQ11_CANDIDATE_MACRO_DICE_DELTAS = (
    0.001136112933560011,
    0.000035058287334012915,
)
V7_REQ11_PRIORITIES = (
    "topology_continuity",
    "deep_grey_boundary",
    "external_csf_retention",
    "tissue_complementarity",
)


@dataclass(frozen=True)
class CandidateEvidence:
    experiment_id: str
    search_type: SearchType
    configuration: dict[str, Any]
    trajectory_identity: str
    fidelity: int
    rung_score: float
    best_score: float
    trajectory_slope: float


@dataclass(frozen=True)
class PortfolioPolicy:
    screening: dict[SearchType, int]
    promotion_targets: dict[int, int]
    wildcard_counts: dict[int, int]
    direct_screening_configurations: tuple[dict[str, Any], ...]

    @classmethod
    def from_runtime(cls, context: TaskRuntimeContext) -> "PortfolioPolicy | None":
        raw = context.task_options.get("campaign_portfolio")
        if raw is None:
            return None
        if not isinstance(raw, dict) or raw.get("version") != PORTFOLIO_VERSION:
            raise ValueError("feta_unet_campaign_portfolio_invalid")
        try:
            screening = {
                SearchType(name): int(value)
                for name, value in dict(raw["screening"]).items()
            }
            promotions = {
                int(name): int(value)
                for name, value in dict(raw["promotion_targets"]).items()
            }
            wildcards = {
                int(name): int(value)
                for name, value in dict(raw["wildcard_counts"]).items()
            }
            direct = tuple(
                dict(item) for item in raw["direct_screening_configurations"]
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("feta_unet_campaign_portfolio_invalid") from exc
        if (
            screening
            != {
                SearchType.OPTUNA: 36,
                SearchType.OPENEVOLVE: 12,
                SearchType.DIRECT: 12,
            }
            or promotions != {50: 18, 100: 7, 150: 2}
            or wildcards != {50: 2, 100: 1, 150: 0}
            or len(direct) < screening[SearchType.DIRECT]
        ):
            raise ValueError("feta_unet_campaign_portfolio_invalid")
        validated: list[dict[str, Any]] = []
        identities: set[str] = set()
        for item in direct:
            candidate = FeTAUNetSearchConfiguration.model_validate(item)
            if candidate.maximum_epochs != 25:
                raise ValueError("feta_unet_campaign_direct_screening_fidelity_invalid")
            identity = trajectory_identity(candidate)
            if identity in identities:
                raise ValueError("feta_unet_campaign_direct_screening_duplicate")
            identities.add(identity)
            validated.append(
                {
                    name: candidate.model_dump(mode="json")[name]
                    for name in CANDIDATE_CONFIGURATION_FIELDS
                }
            )
        return cls(
            screening=screening,
            promotion_targets=promotions,
            wildcard_counts=wildcards,
            direct_screening_configurations=tuple(validated),
        )


@dataclass(frozen=True)
class TreePortfolioPolicy:
    version: str
    root_screening: dict[SearchType, int]
    root_model_variants: dict[str, int]
    root_parent_count: int
    children_per_parent: dict[SearchType, int]
    child_parent_count: int
    grandchildren_per_parent: dict[SearchType, int]
    promotion_targets: dict[int, int]
    wildcard_counts: dict[int, int]
    direct_root_configurations: tuple[dict[str, Any], ...]

    @classmethod
    def from_runtime(cls, context: TaskRuntimeContext) -> "TreePortfolioPolicy":
        raw = context.task_options.get("campaign_portfolio")
        if not isinstance(raw, dict) or raw.get("version") not in {
            TREE_PORTFOLIO_VERSION,
            V6_TREE_PORTFOLIO_VERSION,
        }:
            raise ValueError("feta_unet_campaign_tree_portfolio_invalid")
        version = str(raw["version"])
        try:
            roots = {
                SearchType(name): int(value)
                for name, value in dict(raw["root_screening"]).items()
            }
            children = {
                SearchType(name): int(value)
                for name, value in dict(raw["children_per_parent"]).items()
            }
            grandchildren = {
                SearchType(name): int(value)
                for name, value in dict(raw["grandchildren_per_parent"]).items()
            }
            promotions = {
                int(name): int(value)
                for name, value in dict(raw["promotion_targets"]).items()
            }
            wildcards = {
                int(name): int(value)
                for name, value in dict(raw["wildcard_counts"]).items()
            }
            root_parent_count = int(raw["root_parent_count"])
            child_parent_count = int(raw["child_parent_count"])
            root_model_variants = {
                str(name): int(value)
                for name, value in dict(raw["root_model_variants"]).items()
            }
            direct = tuple(dict(item) for item in raw["direct_root_configurations"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("feta_unet_campaign_tree_portfolio_invalid") from exc
        expected = (
            {
                "roots": {
                    SearchType.OPTUNA: 8,
                    SearchType.OPENEVOLVE: 8,
                    SearchType.DIRECT: 8,
                },
                "root_parent_count": 6,
                "root_model_variants": {
                    "basic_unet": 4,
                    "unet_plain": 2,
                    "unet_residual": 2,
                },
                "child_parent_count": 3,
                "promotions": {50: 8, 100: 4, 150: 2},
                "wildcards": {50: 3, 100: 1, 150: 1},
            }
            if version == TREE_PORTFOLIO_VERSION
            else {
                "roots": {
                    SearchType.OPTUNA: 4,
                    SearchType.OPENEVOLVE: 4,
                    SearchType.DIRECT: 4,
                },
                "root_parent_count": 6,
                "root_model_variants": {"basic_unet": 4},
                "child_parent_count": 3,
                "promotions": {50: 10, 100: 5, 150: 3},
                "wildcards": {50: 3, 100: 2, 150: 1},
            }
        )
        if (
            roots != expected["roots"]
            or root_parent_count != expected["root_parent_count"]
            or root_model_variants != expected["root_model_variants"]
            or children
            != {
                SearchType.OPTUNA: 1,
                SearchType.OPENEVOLVE: 1,
                SearchType.DIRECT: 1,
            }
            or child_parent_count != expected["child_parent_count"]
            or grandchildren != {SearchType.OPTUNA: 1, SearchType.OPENEVOLVE: 1}
            or promotions != expected["promotions"]
            or wildcards != expected["wildcards"]
            or len(direct) < roots[SearchType.DIRECT]
        ):
            raise ValueError("feta_unet_campaign_tree_portfolio_invalid")
        validated: list[dict[str, Any]] = []
        identities: set[str] = set()
        for item in direct:
            candidate = FeTAUNetSearchConfiguration.model_validate(item)
            if candidate.maximum_epochs != 25:
                raise ValueError("feta_unet_campaign_tree_root_fidelity_invalid")
            identity = trajectory_identity(candidate)
            if identity in identities:
                raise ValueError("feta_unet_campaign_tree_root_duplicate")
            identities.add(identity)
            validated.append(
                {
                    name: candidate.model_dump(mode="json")[name]
                    for name in CANDIDATE_CONFIGURATION_FIELDS
                }
            )
        return cls(
            version=version,
            root_screening=roots,
            root_model_variants=root_model_variants,
            root_parent_count=root_parent_count,
            children_per_parent=children,
            child_parent_count=child_parent_count,
            grandchildren_per_parent=grandchildren,
            promotion_targets=promotions,
            wildcard_counts=wildcards,
            direct_root_configurations=tuple(validated),
        )


@dataclass(frozen=True)
class V7MechanismPortfolioPolicy:
    structural_roots: tuple[dict[str, Any], ...]
    mutations_per_root: int
    local_parent_count: int
    optuna_trials_per_parent: int
    wildcard_count: int
    promotion_targets: dict[int, int]
    wildcard_counts: dict[int, int]
    v6_parent_evidence: tuple[dict[str, Any], ...]
    req11_diagnostic: dict[str, Any]

    @classmethod
    def from_runtime(cls, context: TaskRuntimeContext) -> "V7MechanismPortfolioPolicy":
        raw = context.task_options.get("campaign_portfolio")
        if (
            not isinstance(raw, dict)
            or raw.get("version") != V7_MECHANISM_PORTFOLIO_VERSION
        ):
            raise ValueError("feta_unet_v7_portfolio_invalid")
        try:
            roots = tuple(dict(item) for item in raw["structural_roots"])
            mutations_per_root = int(raw["mutations_per_root"])
            local_parent_count = int(raw["local_parent_count"])
            optuna_trials_per_parent = int(raw["optuna_trials_per_parent"])
            wildcard_count = int(raw["controlled_wildcard_count"])
            promotions = {
                int(name): int(value)
                for name, value in dict(raw["promotion_targets"]).items()
            }
            wildcards = {
                int(name): int(value)
                for name, value in dict(raw["wildcard_counts"]).items()
            }
            raw_v6_parents = tuple(dict(item) for item in raw["v6_parent_evidence"])
            req11_diagnostic = dict(raw["req11_diagnostic"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("feta_unet_v7_portfolio_invalid") from exc
        if (
            len(roots) != 4
            or mutations_per_root != 3
            or local_parent_count != 4
            or optuna_trials_per_parent != 2
            or wildcard_count != 2
            or promotions != {50: 8, 100: 4, 150: 2}
            or wildcards != {50: 2, 100: 1, 150: 0}
            or len(raw_v6_parents) != 2
        ):
            raise ValueError("feta_unet_v7_portfolio_invalid")
        try:
            req11_candidates = tuple(
                dict(item) for item in req11_diagnostic["candidates"]
            )
            req11_candidate_ids = tuple(
                str(item["experiment_id"]) for item in req11_candidates
            )
            req11_candidate_deltas = tuple(
                float(item["mean_macro_dice_delta"]) for item in req11_candidates
            )
            req11_priorities = tuple(
                str(item) for item in req11_diagnostic["priorities"]
            )
            complementarity = dict(req11_diagnostic["complementarity"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("feta_unet_v7_req11_diagnostic_invalid") from exc
        if (
            req11_diagnostic.get("schema_version")
            != V7_REQ11_DIAGNOSTIC_SCHEMA_VERSION
            or req11_diagnostic.get("diagnostic_id")
            != "feta-unet-v7-parent-diagnostics-20260822"
            or req11_diagnostic.get("panel_identity") != V7_REQ11_PANEL_IDENTITY
            or req11_diagnostic.get("case_count") != 12
            or req11_diagnostic.get("baseline_experiment_id")
            != V7_REQ11_BASELINE_EXPERIMENT_ID
            or req11_candidate_ids != V7_REQ11_CANDIDATE_EXPERIMENT_IDS
            or any(not math.isfinite(value) for value in req11_candidate_deltas)
            or req11_candidate_deltas != V7_REQ11_CANDIDATE_MACRO_DICE_DELTAS
            or req11_priorities != V7_REQ11_PRIORITIES
            or complementarity
            != {
                "left_material_win_count": 10,
                "right_material_win_count": 17,
                "near_tie_count": 57,
                "observed": True,
            }
            or req11_diagnostic.get("interpretation_boundary")
            != "diagnostic_observation_not_objective"
        ):
            raise ValueError("feta_unet_v7_req11_diagnostic_invalid")
        validated: list[dict[str, Any]] = []
        identities: set[str] = set()
        for item in roots:
            candidate = FeTAUNetSearchConfiguration.model_validate(item)
            if (
                candidate.maximum_epochs != 25
                or candidate.architecture_budget != V7_ARCHITECTURE_BUDGET
                or candidate.model_variant != "structural_basic_unet"
            ):
                raise ValueError("feta_unet_v7_structural_root_invalid")
            identity = trajectory_identity(candidate)
            if identity in identities:
                raise ValueError("feta_unet_v7_structural_root_duplicate")
            identities.add(identity)
            validated.append(
                {
                    name: candidate.model_dump(mode="json")[name]
                    for name in CANDIDATE_CONFIGURATION_FIELDS
                }
            )
        validated_v6_parents: list[dict[str, Any]] = []
        parent_trajectories: set[str] = set()
        for item in raw_v6_parents:
            try:
                experiment_id = str(item["experiment_id"])
                score = float(item["best_score"])
                candidate = FeTAUNetSearchConfiguration.model_validate(
                    item["configuration"]
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("feta_unet_v7_parent_evidence_invalid") from exc
            identity = trajectory_identity(candidate)
            if (
                not experiment_id.startswith("experiment-")
                or not math.isfinite(score)
                or not 0.0 <= score <= 1.0
                or candidate.maximum_epochs != 150
                or candidate.architecture_budget != V6_ARCHITECTURE_BUDGET
                or candidate.model_variant != "basic_unet"
                or identity in parent_trajectories
            ):
                raise ValueError("feta_unet_v7_parent_evidence_invalid")
            parent_trajectories.add(identity)
            validated_v6_parents.append(
                {
                    "experiment_id": experiment_id,
                    "best_score": score,
                    "trajectory_identity": identity,
                    "configuration": candidate.model_dump(mode="json"),
                }
            )
        return cls(
            structural_roots=tuple(validated),
            mutations_per_root=mutations_per_root,
            local_parent_count=local_parent_count,
            optuna_trials_per_parent=optuna_trials_per_parent,
            wildcard_count=wildcard_count,
            promotion_targets=promotions,
            wildcard_counts=wildcards,
            v6_parent_evidence=tuple(validated_v6_parents),
            req11_diagnostic=req11_diagnostic,
        )


@dataclass(frozen=True)
class TreeCandidate:
    evidence: CandidateEvidence
    stage: str
    action: SearchType
    parent_trajectory: str | None
    root_trajectory: str


def _outputs(event: DecisionEvent) -> dict[str, str]:
    return {
        key: value
        for reference in event.output_references
        if ":" in reference
        for key, value in (reference.split(":", 1),)
    }


def _evidence(events: tuple[DecisionEvent, ...]) -> tuple[CandidateEvidence, ...]:
    rows: list[CandidateEvidence] = []
    for event in events:
        if event.event_type != EventType.EVIDENCE_VERIFIED:
            continue
        values = _outputs(event)
        if (
            values.get("verified") != "true"
            or values.get("constraints") != "true"
            or values.get("evidence") != EvidenceStatus.SUPPORTED.value
            or not event.input_references
        ):
            continue
        try:
            search_type = SearchType(values["search_type"])
            configuration = FeTAUNetSearchConfiguration.model_validate(
                dict(event.safe_payload["configuration"])
            )
            score = float(values["score"])
        except (KeyError, TypeError, ValueError):
            continue
        aggregate = event.safe_payload.get("aggregate_metrics", {})
        history = (
            aggregate.get("validation_history", ())
            if isinstance(aggregate, dict)
            else ()
        )
        entries = [item for item in history if isinstance(item, dict)]
        endpoint = next(
            (
                item
                for item in reversed(entries)
                if item.get("epoch") == configuration.maximum_epochs
            ),
            None,
        )
        rung_score = score
        if endpoint is not None and isinstance(
            endpoint.get("validation_score"), (int, float)
        ):
            rung_score = float(endpoint["validation_score"])
        slope = 0.0
        if len(entries) >= 2:
            first, last = entries[0], entries[-1]
            if all(
                isinstance(item, (int, float))
                for item in (
                    first.get("epoch"),
                    first.get("validation_score"),
                    last.get("epoch"),
                    last.get("validation_score"),
                )
            ) and float(last["epoch"]) > float(first["epoch"]):
                slope = (
                    float(last["validation_score"]) - float(first["validation_score"])
                ) / (float(last["epoch"]) - float(first["epoch"]))
        rows.append(
            CandidateEvidence(
                experiment_id=event.input_references[0],
                search_type=search_type,
                configuration={
                    name: configuration.model_dump(mode="json")[name]
                    for name in CANDIDATE_CONFIGURATION_FIELDS
                },
                trajectory_identity=trajectory_identity(configuration),
                fidelity=configuration.maximum_epochs,
                rung_score=rung_score,
                best_score=score,
                trajectory_slope=slope,
            )
        )
    return tuple(rows)


def _unique_at_fidelity(
    rows: tuple[CandidateEvidence, ...], fidelity: int
) -> dict[str, CandidateEvidence]:
    selected: dict[str, CandidateEvidence] = {}
    for row in rows:
        if row.fidelity != fidelity:
            continue
        existing = selected.get(row.trajectory_identity)
        if existing is None or (row.rung_score, row.experiment_id) > (
            existing.rung_score,
            existing.experiment_id,
        ):
            selected[row.trajectory_identity] = row
    return selected


def _promotion_cohort(
    source: dict[str, CandidateEvidence],
    *,
    target: int,
    wildcard_count: int,
) -> tuple[CandidateEvidence, ...]:
    ranked = sorted(
        source.values(),
        key=lambda item: (item.rung_score, item.trajectory_identity),
        reverse=True,
    )
    if len(ranked) < target:
        raise ValueError("feta_unet_campaign_promotion_cohort_incomplete")
    ranked_count = target - wildcard_count
    selected = list(ranked[:ranked_count])
    selected_ids = {item.trajectory_identity for item in selected}
    remaining = [
        item
        for item in ranked[ranked_count:]
        if item.trajectory_identity not in selected_ids
    ]
    represented = {item.search_type for item in selected}
    for method in (SearchType.OPTUNA, SearchType.OPENEVOLVE, SearchType.DIRECT):
        if len(selected) >= target:
            break
        if method in represented:
            continue
        candidate = next(
            (item for item in remaining if item.search_type == method), None
        )
        if candidate is not None:
            selected.append(candidate)
            selected_ids.add(candidate.trajectory_identity)
            remaining.remove(candidate)
    remaining.sort(
        key=lambda item: (
            item.trajectory_slope,
            item.rung_score,
            item.trajectory_identity,
        ),
        reverse=True,
    )
    for item in remaining:
        if len(selected) >= target:
            break
        if item.trajectory_identity not in selected_ids:
            selected.append(item)
            selected_ids.add(item.trajectory_identity)
    return tuple(selected)


def _request(
    original: SearchRequest,
    *,
    run_id: str,
    cycle: int,
    stage: str,
    search_type: SearchType,
    search_space: dict[str, Any],
    experiment_budget: int,
    rationale: str,
    evidence_references: tuple[str, ...] = (),
) -> SearchRequest:
    digest = hashlib.sha256(
        payload_hash(
            {
                "run_id": run_id,
                "cycle": cycle,
                "stage": stage,
                "search_type": search_type.value,
                "search_space": search_space,
                "experiment_budget": experiment_budget,
            }
        ).encode()
    ).hexdigest()[:20]
    return SearchRequest(
        request_id=f"search-portfolio-{digest}",
        hypothesis_id=original.hypothesis_id,
        search_type=search_type,
        target=original.target,
        search_space=search_space,
        experiment_budget=experiment_budget,
        rationale=rationale,
        evidence_references=evidence_references,
        requires_human_approval=False,
        proposal_source=ProposalSource.DETERMINISTIC,
        grounding_status=original.grounding_status,
        agent_call_id=original.agent_call_id,
        prompt_version=original.prompt_version,
    )


def _tree_request_metadata(
    events: tuple[DecisionEvent, ...],
) -> dict[str, dict[str, str]]:
    def metadata_from_references(references: tuple[str, ...]) -> dict[str, str]:
        metadata: dict[str, str] = {}
        for reference in references:
            value = reference.removeprefix("evidence_reference:")
            if value == reference:
                continue
            for name in (
                "tree-stage",
                "tree-action",
                "tree-parent",
                "tree-root",
            ):
                marker = f"{name}:"
                if value.startswith(marker):
                    metadata[name] = value[len(marker) :]
        return metadata

    requests: dict[str, dict[str, str]] = {}
    missing_requests: set[str] = set()
    for event in events:
        if event.event_type != EventType.SEARCH_PLANNED:
            continue
        request_id = next(
            (item for item in event.output_references if ":" not in item),
            None,
        )
        if request_id is None:
            continue
        metadata = metadata_from_references(event.output_references)
        if metadata.get("tree-stage"):
            requests[request_id] = metadata
        elif event.rationale.startswith(f"{TREE_PORTFOLIO_VERSION}:"):
            missing_requests.add(request_id)
    openevolve_prepared_experiments = {
        event.output_references[0]
        for event in events
        if event.event_type == EventType.OPENEVOLVE_CANDIDATE_PREPARED
        and event.output_references
    }
    experiments: dict[str, dict[str, str]] = {}
    for event in events:
        if (
            event.event_type != EventType.EXPERIMENT_PREPARED
            or not event.input_references
            or not event.output_references
        ):
            continue
        request_metadata = requests.get(event.input_references[0])
        embedded_metadata = metadata_from_references(event.output_references)
        if (
            request_metadata is not None
            and embedded_metadata
            and request_metadata != embedded_metadata
        ):
            experiment_id = event.output_references[0]
            matching_reuse_requests = tuple(
                request_id
                for request_id, metadata in requests.items()
                if metadata == embedded_metadata
            )
            canonical_openevolve_reuse = (
                experiment_id in openevolve_prepared_experiments
                and request_metadata.get("tree-action") != SearchType.OPENEVOLVE.value
                and embedded_metadata.get("tree-action") == SearchType.OPENEVOLVE.value
                and len(matching_reuse_requests) == 1
            )
            if not canonical_openevolve_reuse:
                raise ValueError("feta_unet_campaign_tree_metadata_conflict")
            # This is a legacy provenance shape produced when OpenEvolve reused
            # a canonical generation-zero experiment.  Keep the experiment's
            # original method ownership, while marking the OpenEvolve request
            # as consumed by the explicitly recorded reuse operation.
            missing_requests.discard(matching_reuse_requests[0])
            metadata = request_metadata
        else:
            metadata = embedded_metadata or request_metadata
        if metadata is not None:
            experiments[event.output_references[0]] = metadata
            missing_requests.discard(event.input_references[0])
    if missing_requests:
        raise ValueError("feta_unet_campaign_tree_metadata_missing")
    return experiments


def _tree_candidates(
    events: tuple[DecisionEvent, ...], rows: tuple[CandidateEvidence, ...]
) -> tuple[TreeCandidate, ...]:
    metadata = _tree_request_metadata(events)
    candidates: list[TreeCandidate] = []
    for row in rows:
        item = metadata.get(row.experiment_id)
        if item is None:
            continue
        try:
            action = SearchType(item["tree-action"])
            stage = item["tree-stage"]
        except (KeyError, ValueError):
            continue
        parent = item.get("tree-parent")
        root = (
            row.trajectory_identity
            if stage == "root"
            else item.get("tree-root") or row.trajectory_identity
        )
        candidates.append(
            TreeCandidate(
                evidence=row,
                stage=stage,
                action=action,
                parent_trajectory=parent,
                root_trajectory=root,
            )
        )
    seen: dict[tuple[str, SearchType, str], str] = {}
    for candidate in candidates:
        key = (
            candidate.stage,
            candidate.action,
            candidate.evidence.trajectory_identity,
        )
        previous = seen.setdefault(key, candidate.evidence.experiment_id)
        if previous != candidate.evidence.experiment_id:
            raise ValueError("feta_unet_campaign_tree_duplicate_execution")
    return tuple(candidates)


def _unique_tree_stage(
    candidates: tuple[TreeCandidate, ...], stage: str
) -> dict[str, TreeCandidate]:
    selected: dict[str, TreeCandidate] = {}
    for candidate in candidates:
        if candidate.stage != stage:
            continue
        identity = candidate.evidence.trajectory_identity
        existing = selected.get(identity)
        if existing is None or (
            candidate.evidence.best_score,
            candidate.evidence.rung_score,
            candidate.evidence.experiment_id,
        ) > (
            existing.evidence.best_score,
            existing.evidence.rung_score,
            existing.evidence.experiment_id,
        ):
            selected[identity] = candidate
    return selected


def _tree_cohort(
    candidates: tuple[TreeCandidate, ...],
    *,
    target: int,
    wildcard_count: int,
) -> tuple[TreeCandidate, ...]:
    unique = {item.evidence.trajectory_identity: item for item in candidates}
    ranked = sorted(
        unique.values(),
        key=lambda item: (
            item.evidence.best_score,
            item.evidence.rung_score,
            item.evidence.trajectory_slope,
            item.evidence.trajectory_identity,
        ),
        reverse=True,
    )
    if len(ranked) < target:
        raise ValueError("feta_unet_campaign_tree_cohort_incomplete")
    selected = list(ranked[: target - wildcard_count])
    selected_ids = {item.evidence.trajectory_identity for item in selected}
    remaining = [
        item for item in ranked if item.evidence.trajectory_identity not in selected_ids
    ]
    represented_methods = {item.action for item in selected}
    represented_roots = {item.root_trajectory for item in selected}
    for method in (SearchType.OPTUNA, SearchType.OPENEVOLVE, SearchType.DIRECT):
        if len(selected) >= target or method in represented_methods:
            continue
        candidate = next((item for item in remaining if item.action == method), None)
        if candidate is not None:
            selected.append(candidate)
            remaining.remove(candidate)
            selected_ids.add(candidate.evidence.trajectory_identity)
            represented_methods.add(method)
            represented_roots.add(candidate.root_trajectory)
    remaining.sort(
        key=lambda item: (
            item.root_trajectory not in represented_roots,
            item.evidence.trajectory_slope,
            item.evidence.best_score,
            item.evidence.trajectory_identity,
        ),
        reverse=True,
    )
    for item in remaining:
        if len(selected) >= target:
            break
        selected.append(item)
        represented_roots.add(item.root_trajectory)
    return tuple(selected)


def _tree_references(
    *,
    stage: str,
    action: SearchType,
    parent: str | None = None,
    root: str | None = None,
    extra: tuple[str, ...] = (),
) -> tuple[str, ...]:
    return (
        f"tree-stage:{stage}",
        f"tree-action:{action.value}",
        *((f"tree-parent:{parent}",) if parent is not None else ()),
        *((f"tree-root:{root}",) if root is not None else ()),
        *extra,
    )


def _local_optuna_space(
    parent: CandidateEvidence, *, seed: int, trial_budget: int
) -> dict[str, Any]:
    configuration = parent.configuration
    learning_rate = float(configuration["learning_rate"])
    weight_decay = float(configuration["weight_decay"])
    dropout = float(configuration["dropout"])
    dice_weight = float(configuration["dice_weight"])
    return {
        "seed": seed,
        "n_startup_trials": min(2, trial_budget),
        "fixed": {
            "maximum_epochs": 25,
            **{
                name: configuration[name]
                for name in (
                    "model_variant",
                    "feature_width",
                    "features",
                    "architecture_budget",
                    "upsample",
                    "kernel_profile",
                    "residual_blocks",
                    "deep_supervision_heads",
                    "convolutions_per_stage",
                    "dilation_profile",
                    "skip_fusion",
                    "downsample",
                    "activation",
                    "norm",
                    "optimizer",
                )
            },
        },
        "parameters": {
            "learning_rate": {
                "low": max(LEARNING_RATE_BOUNDS[0], learning_rate / 2.0),
                "high": min(LEARNING_RATE_BOUNDS[1], learning_rate * 2.0),
            },
            "weight_decay": {
                "low": max(WEIGHT_DECAY_BOUNDS[0], weight_decay / 3.0),
                "high": min(WEIGHT_DECAY_BOUNDS[1], weight_decay * 3.0),
            },
            "dropout": {
                "low": max(DROPOUT_BOUNDS[0], dropout - 0.08),
                "high": min(DROPOUT_BOUNDS[1], dropout + 0.08),
            },
            "dice_weight": {
                "low": max(DICE_WEIGHT_BOUNDS[0], dice_weight - 0.2),
                "high": min(DICE_WEIGHT_BOUNDS[1], dice_weight + 0.2),
            },
            "positive_negative_ratio": {"choices": ["1:1", "2:1", "3:1"]},
            "lr_schedule": {"choices": ["constant", "cosine", "polynomial"]},
            "loss_variant": {"choices": ["dice_ce", "dice_focal", "dice_tversky"]},
            "augmentation_policy": {
                "choices": [
                    "reference_light",
                    "geometric",
                    "intensity",
                    "combined",
                ]
            },
        },
    }


def _direct_ablation(
    parent: CandidateEvidence,
    existing: set[str],
    *,
    allowed_model_variants: tuple[str, ...] = MODEL_VARIANTS,
) -> dict[str, Any]:
    base = {name: parent.configuration[name] for name in CANDIDATE_CONFIGURATION_FIELDS}
    feature_widths = (
        tuple(V6_BASIC_UNET_FEATURE_PROFILES)
        if base.get("architecture_budget") == V6_ARCHITECTURE_BUDGET
        else ("narrow", "baseline", "wide")
    )
    axes: tuple[tuple[str, tuple[Any, ...]], ...] = (
        ("model_variant", allowed_model_variants),
        ("lr_schedule", ("constant", "cosine", "polynomial")),
        ("optimizer", ("AdamW", "Adam")),
        ("loss_variant", LOSS_VARIANTS),
        ("norm", ("instance", "group")),
        ("activation", ("LeakyReLU", "ReLU", "PReLU")),
        ("feature_width", feature_widths),
        ("augmentation_policy", AUGMENTATION_POLICIES),
    )
    for name, values in axes:
        for value in values:
            if value == base[name]:
                continue
            candidate = {**base, name: value, "maximum_epochs": 25}
            if name == "feature_width":
                candidate.pop("features", None)
            validated = FeTAUNetSearchConfiguration.model_validate(candidate)
            if trajectory_identity(validated) not in existing:
                return {
                    key: validated.model_dump(mode="json")[key]
                    for key in CANDIDATE_CONFIGURATION_FIELDS
                }
    raise ValueError("feta_unet_campaign_tree_direct_ablation_exhausted")


def _tree_seed(parent: str, action: SearchType, completed: int) -> int:
    digest = hashlib.sha256(
        f"{parent}\x1f{action.value}\x1f{completed}".encode()
    ).hexdigest()
    return int(digest[:8], 16)


def apply_tree_portfolio_policy(
    original: SearchRequest,
    *,
    run_id: str,
    cycle: int,
    events: tuple[DecisionEvent, ...],
    runtime_context: TaskRuntimeContext,
) -> SearchRequest | None:
    policy = TreePortfolioPolicy.from_runtime(runtime_context)
    rows = _evidence(events)
    candidates = _tree_candidates(events, rows)

    root_candidates = tuple(item for item in candidates if item.stage == "root")
    root_by_method: dict[SearchType, dict[str, TreeCandidate]] = {}
    prior_root_ids: set[str] = set()
    for method in (SearchType.OPTUNA, SearchType.OPENEVOLVE, SearchType.DIRECT):
        selected: dict[str, TreeCandidate] = {}
        for item in root_candidates:
            identity = item.evidence.trajectory_identity
            if item.action != method or identity in prior_root_ids:
                continue
            selected.setdefault(identity, item)
        root_by_method[method] = selected
        prior_root_ids.update(selected)

    optuna_roots = root_by_method[SearchType.OPTUNA]
    for model_variant, target in policy.root_model_variants.items():
        completed = {
            identity
            for identity, item in optuna_roots.items()
            if item.evidence.configuration["model_variant"] == model_variant
        }
        if len(completed) >= target:
            continue
        remaining = target - len(completed)
        v6_architecture = policy.version == V6_TREE_PORTFOLIO_VERSION
        fixed = {
            "maximum_epochs": 25,
            "model_variant": model_variant,
        }
        parameters: dict[str, Any] = {}
        if v6_architecture:
            fixed["architecture_budget"] = V6_ARCHITECTURE_BUDGET
            fixed["upsample"] = "deconv"
            parameters["feature_width"] = {"choices": list(V6_OPTUNA_FEATURE_PROFILES)}
        return _request(
            original,
            run_id=run_id,
            cycle=cycle,
            stage="tree-root-optuna",
            search_type=SearchType.OPTUNA,
            search_space={
                "seed": _tree_seed(
                    f"{run_id}:root:{model_variant}", SearchType.OPTUNA, 0
                ),
                "fixed": fixed,
                **({"parameters": parameters} if parameters else {}),
            },
            experiment_budget=remaining,
            rationale=(
                f"{TREE_PORTFOLIO_VERSION}: create {remaining} deduplicated {model_variant} Optuna root lineages at 25 epochs."
            ),
            evidence_references=_tree_references(
                stage="root", action=SearchType.OPTUNA
            ),
        )

    oe_roots = root_by_method[SearchType.OPENEVOLVE]
    for model_variant, target in policy.root_model_variants.items():
        completed = {
            identity
            for identity, item in oe_roots.items()
            if item.evidence.configuration["model_variant"] == model_variant
        }
        if len(completed) >= target:
            continue
        remaining = target - len(completed)
        evaluations = remaining + 1
        parent = max(
            (
                item
                for item in optuna_roots.values()
                if item.evidence.configuration["model_variant"] == model_variant
            ),
            key=lambda item: (
                item.evidence.best_score,
                item.evidence.trajectory_identity,
            ),
        )
        search_space = default_openevolve_configuration(
            candidate_evaluations=evaluations
        )
        search_space["openevolve"].update(
            {
                "maximum_failed_candidates": evaluations,
                "maximum_consecutive_failures": evaluations,
            }
        )
        search_space["campaign_context"] = {
            "incumbent_training_policy": policy_from_configuration(
                parent.evidence.configuration
            ).model_dump(mode="json"),
            "incumbent_primary_score": parent.evidence.best_score,
            "incumbent_search_type": parent.action.value,
            "incumbent_experiment_id": parent.evidence.experiment_id,
            "required_model_variant": model_variant,
            **(
                {"required_architecture_budget": V6_ARCHITECTURE_BUDGET}
                if policy.version == V6_TREE_PORTFOLIO_VERSION
                else {}
            ),
            "prior_verified_results": [
                {
                    "search_type": item.action.value,
                    "primary_score": item.evidence.best_score,
                    "configuration": item.evidence.configuration,
                }
                for item in sorted(
                    (
                        candidate
                        for candidate in optuna_roots.values()
                        if candidate.evidence.configuration["model_variant"]
                        == model_variant
                    ),
                    key=lambda item: item.evidence.best_score,
                    reverse=True,
                )
            ],
        }
        return _request(
            original,
            run_id=run_id,
            cycle=cycle,
            stage="tree-root-openevolve",
            search_type=SearchType.OPENEVOLVE,
            search_space=search_space,
            experiment_budget=evaluations,
            rationale=(
                f"{TREE_PORTFOLIO_VERSION}: create {remaining} novel {model_variant} OpenEvolve roots from the strongest matching Optuna anchor."
            ),
            evidence_references=_tree_references(
                stage="root",
                action=SearchType.OPENEVOLVE,
                parent=parent.evidence.trajectory_identity,
                root=parent.evidence.trajectory_identity,
            ),
        )

    direct_roots = root_by_method[SearchType.DIRECT]
    for model_variant, target in policy.root_model_variants.items():
        completed = {
            identity
            for identity, item in direct_roots.items()
            if item.evidence.configuration["model_variant"] == model_variant
        }
        if len(completed) >= target:
            continue
        existing = set().union(*(set(items) for items in root_by_method.values()))
        configuration = next(
            (
                item
                for item in policy.direct_root_configurations
                if item["model_variant"] == model_variant
                if trajectory_identity(FeTAUNetSearchConfiguration.model_validate(item))
                not in existing
            ),
            None,
        )
        if configuration is None:
            raise ValueError("feta_unet_campaign_tree_direct_root_pool_exhausted")
        return _request(
            original,
            run_id=run_id,
            cycle=cycle,
            stage="tree-root-direct",
            search_type=SearchType.DIRECT,
            search_space=configuration,
            experiment_budget=1,
            rationale=(
                f"{TREE_PORTFOLIO_VERSION}: execute one deduplicated {model_variant} DIRECT root ablation."
            ),
            evidence_references=_tree_references(
                stage="root", action=SearchType.DIRECT
            ),
        )

    roots = tuple(
        item for method in root_by_method.values() for item in method.values()
    )
    root_parents = _tree_cohort(
        roots,
        target=policy.root_parent_count,
        wildcard_count=2,
    )
    root_ids = {item.evidence.trajectory_identity for item in roots}
    child_candidates = tuple(
        item
        for item in candidates
        if item.stage == "child" and item.evidence.trajectory_identity not in root_ids
    )
    existing_25 = {
        item.evidence.trajectory_identity for item in (*roots, *child_candidates)
    }
    for parent in root_parents:
        parent_id = parent.evidence.trajectory_identity
        for action in (SearchType.OPTUNA, SearchType.OPENEVOLVE, SearchType.DIRECT):
            target = policy.children_per_parent[action]
            completed = {
                item.evidence.trajectory_identity
                for item in child_candidates
                if item.parent_trajectory == parent_id and item.action == action
            }
            if len(completed) >= target:
                continue
            remaining = target - len(completed)
            references = _tree_references(
                stage="child",
                action=action,
                parent=parent_id,
                root=parent.root_trajectory,
            )
            if action == SearchType.OPTUNA:
                search_space = _local_optuna_space(
                    parent.evidence,
                    seed=_tree_seed(parent_id, action, len(completed)),
                    trial_budget=remaining,
                )
                return _request(
                    original,
                    run_id=run_id,
                    cycle=cycle,
                    stage="tree-child-optuna",
                    search_type=action,
                    search_space=search_space,
                    experiment_budget=remaining,
                    rationale=(
                        f"{TREE_PORTFOLIO_VERSION}: refine root {parent_id[:12]} with {remaining} local Optuna children."
                    ),
                    evidence_references=references,
                )
            if action == SearchType.OPENEVOLVE:
                evaluations = remaining + 1
                search_space = default_openevolve_configuration(
                    candidate_evaluations=evaluations
                )
                search_space["openevolve"].update(
                    {
                        "maximum_failed_candidates": evaluations,
                        "maximum_consecutive_failures": evaluations,
                    }
                )
                search_space["campaign_context"] = {
                    "incumbent_training_policy": policy_from_configuration(
                        parent.evidence.configuration
                    ).model_dump(mode="json"),
                    "incumbent_primary_score": parent.evidence.best_score,
                    "incumbent_search_type": parent.action.value,
                    "incumbent_experiment_id": parent.evidence.experiment_id,
                    "required_model_variant": parent.evidence.configuration[
                        "model_variant"
                    ],
                    **(
                        {"required_architecture_budget": V6_ARCHITECTURE_BUDGET}
                        if policy.version == V6_TREE_PORTFOLIO_VERSION
                        else {}
                    ),
                    "prior_verified_results": [
                        {
                            "search_type": parent.action.value,
                            "primary_score": parent.evidence.best_score,
                            "configuration": parent.evidence.configuration,
                        }
                    ],
                }
                return _request(
                    original,
                    run_id=run_id,
                    cycle=cycle,
                    stage="tree-child-openevolve",
                    search_type=action,
                    search_space=search_space,
                    experiment_budget=evaluations,
                    rationale=(
                        f"{TREE_PORTFOLIO_VERSION}: evolve one bounded child from root {parent_id[:12]}."
                    ),
                    evidence_references=references,
                )
            configuration = _direct_ablation(
                parent.evidence,
                existing_25,
                allowed_model_variants=tuple(policy.root_model_variants),
            )
            return _request(
                original,
                run_id=run_id,
                cycle=cycle,
                stage="tree-child-direct",
                search_type=action,
                search_space=configuration,
                experiment_budget=1,
                rationale=(
                    f"{TREE_PORTFOLIO_VERSION}: execute one controlled child ablation from root {parent_id[:12]}."
                ),
                evidence_references=references,
            )

    children = tuple(_unique_tree_stage(child_candidates, "child").values())
    child_parents = _tree_cohort(
        children,
        target=policy.child_parent_count,
        wildcard_count=2,
    )
    child_ids = {item.evidence.trajectory_identity for item in children}
    grandchild_candidates = tuple(
        item
        for item in candidates
        if item.stage == "grandchild"
        and item.evidence.trajectory_identity not in root_ids | child_ids
    )
    existing_25.update(
        item.evidence.trajectory_identity for item in grandchild_candidates
    )
    for parent in child_parents:
        parent_id = parent.evidence.trajectory_identity
        for action in (SearchType.OPTUNA, SearchType.OPENEVOLVE):
            target = policy.grandchildren_per_parent[action]
            completed = {
                item.evidence.trajectory_identity
                for item in grandchild_candidates
                if item.parent_trajectory == parent_id and item.action == action
            }
            if len(completed) >= target:
                continue
            remaining = target - len(completed)
            references = _tree_references(
                stage="grandchild",
                action=action,
                parent=parent_id,
                root=parent.root_trajectory,
            )
            if action == SearchType.OPTUNA:
                return _request(
                    original,
                    run_id=run_id,
                    cycle=cycle,
                    stage="tree-grandchild-optuna",
                    search_type=action,
                    search_space=_local_optuna_space(
                        parent.evidence,
                        seed=_tree_seed(parent_id, action, len(completed)),
                        trial_budget=remaining,
                    ),
                    experiment_budget=remaining,
                    rationale=(
                        f"{TREE_PORTFOLIO_VERSION}: locally refine child {parent_id[:12]} with one Optuna grandchild."
                    ),
                    evidence_references=references,
                )
            evaluations = remaining + 1
            search_space = default_openevolve_configuration(
                candidate_evaluations=evaluations
            )
            search_space["openevolve"].update(
                {
                    "maximum_failed_candidates": evaluations,
                    "maximum_consecutive_failures": evaluations,
                }
            )
            search_space["campaign_context"] = {
                "incumbent_training_policy": policy_from_configuration(
                    parent.evidence.configuration
                ).model_dump(mode="json"),
                "incumbent_primary_score": parent.evidence.best_score,
                "incumbent_search_type": parent.action.value,
                "incumbent_experiment_id": parent.evidence.experiment_id,
                "required_model_variant": parent.evidence.configuration[
                    "model_variant"
                ],
                **(
                    {"required_architecture_budget": V6_ARCHITECTURE_BUDGET}
                    if policy.version == V6_TREE_PORTFOLIO_VERSION
                    else {}
                ),
                "prior_verified_results": [
                    {
                        "search_type": parent.action.value,
                        "primary_score": parent.evidence.best_score,
                        "configuration": parent.evidence.configuration,
                    }
                ],
            }
            return _request(
                original,
                run_id=run_id,
                cycle=cycle,
                stage="tree-grandchild-openevolve",
                search_type=action,
                search_space=search_space,
                experiment_budget=evaluations,
                rationale=(
                    f"{TREE_PORTFOLIO_VERSION}: evolve one grandchild from child {parent_id[:12]}."
                ),
                evidence_references=references,
            )

    grandchildren = tuple(
        _unique_tree_stage(grandchild_candidates, "grandchild").values()
    )
    source = tuple((*roots, *children, *grandchildren))
    source_fidelity = 25
    for target_fidelity in (50, 100, 150):
        target_count = policy.promotion_targets[target_fidelity]
        cohort = _tree_cohort(
            source,
            target=target_count,
            wildcard_count=policy.wildcard_counts[target_fidelity],
        )
        completed = _unique_tree_stage(candidates, f"promote-{target_fidelity}")
        pending = next(
            (
                item
                for item in cohort
                if item.evidence.trajectory_identity not in completed
            ),
            None,
        )
        if pending is not None:
            configuration = dict(pending.evidence.configuration)
            configuration["maximum_epochs"] = target_fidelity
            return _request(
                original,
                run_id=run_id,
                cycle=cycle,
                stage=f"tree-promote-{source_fidelity}-{target_fidelity}",
                search_type=SearchType.DIRECT,
                search_space=configuration,
                experiment_budget=1,
                rationale=(
                    f"{TREE_PORTFOLIO_VERSION}: promote the {pending.action.value} lineage {pending.evidence.trajectory_identity[:12]} from {source_fidelity} to {target_fidelity} epochs using best-checkpoint and diversity evidence."
                ),
                evidence_references=_tree_references(
                    stage=f"promote-{target_fidelity}",
                    action=pending.action,
                    parent=pending.evidence.trajectory_identity,
                    root=pending.root_trajectory,
                    extra=(
                        pending.evidence.experiment_id,
                        f"promotion-from-epoch:{source_fidelity}",
                        f"origin-search-type:{pending.action.value}",
                    ),
                ),
            )
        source = tuple(completed.values())
        source_fidelity = target_fidelity
    return None


def _v7_controlled_wildcard(
    parent: CandidateEvidence,
    existing: set[str],
) -> dict[str, Any]:
    base = {name: parent.configuration[name] for name in CANDIDATE_CONFIGURATION_FIELDS}
    axes: tuple[tuple[str, tuple[Any, ...]], ...] = (
        ("kernel_profile", ("standard", "large_front", "context_deep")),
        ("residual_blocks", (False, True)),
        ("deep_supervision_heads", (0, 1, 2)),
        ("loss_variant", LOSS_VARIANTS),
        ("positive_negative_ratio", ("1:1", "2:1", "3:1")),
        ("augmentation_policy", AUGMENTATION_POLICIES),
    )
    for name, values in axes:
        for value in values:
            if value == base[name]:
                continue
            candidate = {**base, name: value, "maximum_epochs": 25}
            try:
                validated = FeTAUNetSearchConfiguration.model_validate(candidate)
            except ValueError:
                continue
            if trajectory_identity(validated) not in existing:
                return {
                    key: validated.model_dump(mode="json")[key]
                    for key in CANDIDATE_CONFIGURATION_FIELDS
                }
    raise ValueError("feta_unet_v7_controlled_wildcard_exhausted")


def apply_v7_mechanism_portfolio_policy(
    original: SearchRequest,
    *,
    run_id: str,
    cycle: int,
    events: tuple[DecisionEvent, ...],
    runtime_context: TaskRuntimeContext,
) -> SearchRequest | None:
    """Run mechanism roots, structural evolution, local HPO and graduation."""

    policy = V7MechanismPortfolioPolicy.from_runtime(runtime_context)
    rows = _evidence(events)
    candidates = _tree_candidates(events, rows)
    roots = _unique_tree_stage(
        tuple(
            item
            for item in candidates
            if item.stage == "root" and item.action == SearchType.DIRECT
        ),
        "root",
    )
    for configuration in policy.structural_roots:
        candidate = FeTAUNetSearchConfiguration.model_validate(configuration)
        identity = trajectory_identity(candidate)
        if identity in roots:
            continue
        return _request(
            original,
            run_id=run_id,
            cycle=cycle,
            stage="v7-mechanism-root",
            search_type=SearchType.DIRECT,
            search_space=configuration,
            experiment_budget=1,
            rationale=(
                f"{V7_MECHANISM_PORTFOLIO_VERSION}: execute one frozen structural BasicUNet root with distinct depth, receptive-field, residual, skip and deep-supervision mechanisms."
            ),
            evidence_references=_tree_references(
                stage="root", action=SearchType.DIRECT
            ),
        )

    root_items = tuple(roots.values())
    structural_candidates = tuple(
        item
        for item in candidates
        if item.stage == "structural-child" and item.action == SearchType.OPENEVOLVE
    )
    for parent in root_items:
        parent_id = parent.evidence.trajectory_identity
        completed = {
            item.evidence.trajectory_identity
            for item in structural_candidates
            if item.parent_trajectory == parent_id
            and item.evidence.trajectory_identity != parent_id
        }
        if len(completed) >= policy.mutations_per_root:
            continue
        remaining = policy.mutations_per_root - len(completed)
        evaluations = remaining + 1
        search_space = default_openevolve_configuration(
            candidate_evaluations=evaluations
        )
        search_space["openevolve"].update(
            {
                "maximum_failed_candidates": evaluations,
                "maximum_consecutive_failures": evaluations,
            }
        )
        search_space["campaign_context"] = {
            "incumbent_training_policy": policy_from_configuration(
                parent.evidence.configuration
            ).model_dump(mode="json"),
            "incumbent_primary_score": parent.evidence.best_score,
            "incumbent_search_type": parent.action.value,
            "incumbent_experiment_id": parent.evidence.experiment_id,
            "required_model_variant": "structural_basic_unet",
            "required_architecture_budget": V7_ARCHITECTURE_BUDGET,
            "mutation_objective": (
                "Change at least one structural mechanism among depth or non-uniform stage widths, convolutions per stage, kernel or dilation profile, residual blocks, skip fusion, down/up operator and deep-supervision heads before local optimisation. Use the bound REQ-11 observations to prioritise mechanisms plausibly improving topology continuity, deep-grey boundary quality and external-CSF retention without treating diagnostic metrics as optimisation objectives; preserve diversity because the verified parents showed tissue-level complementarity."
            ),
            "req11_diagnostic_evidence": policy.req11_diagnostic,
            "prior_verified_results": [
                {
                    "search_type": parent.action.value,
                    "primary_score": parent.evidence.best_score,
                    "configuration": parent.evidence.configuration,
                },
                *(
                    {
                        "search_type": "DIRECT",
                        "primary_score": item["best_score"],
                        "configuration": item["configuration"],
                        "source_experiment_id": item["experiment_id"],
                        "trajectory_identity": item["trajectory_identity"],
                        "evidence_role": "v6_parent_not_retrained",
                    }
                    for item in policy.v6_parent_evidence
                ),
            ],
        }
        return _request(
            original,
            run_id=run_id,
            cycle=cycle,
            stage="v7-structural-openevolve",
            search_type=SearchType.OPENEVOLVE,
            search_space=search_space,
            experiment_budget=evaluations,
            rationale=(
                f"{V7_MECHANISM_PORTFOLIO_VERSION}: generate {remaining} novel structural mutations from root {parent_id[:12]}."
            ),
            evidence_references=_tree_references(
                stage="structural-child",
                action=SearchType.OPENEVOLVE,
                parent=parent_id,
                root=parent.root_trajectory,
            ),
        )

    structural = tuple(
        item
        for item in _unique_tree_stage(
            structural_candidates, "structural-child"
        ).values()
        if item.parent_trajectory != item.evidence.trajectory_identity
    )
    local_parents = _tree_cohort(
        structural,
        target=policy.local_parent_count,
        wildcard_count=1,
    )
    local_candidates = tuple(
        item
        for item in candidates
        if item.stage == "local-optuna" and item.action == SearchType.OPTUNA
    )
    for parent in local_parents:
        parent_id = parent.evidence.trajectory_identity
        completed = {
            item.evidence.trajectory_identity
            for item in local_candidates
            if item.parent_trajectory == parent_id
        }
        if len(completed) >= policy.optuna_trials_per_parent:
            continue
        remaining = policy.optuna_trials_per_parent - len(completed)
        return _request(
            original,
            run_id=run_id,
            cycle=cycle,
            stage="v7-lineage-local-optuna",
            search_type=SearchType.OPTUNA,
            search_space=_local_optuna_space(
                parent.evidence,
                seed=_tree_seed(parent_id, SearchType.OPTUNA, len(completed)),
                trial_budget=remaining,
            ),
            experiment_budget=remaining,
            rationale=(
                f"{V7_MECHANISM_PORTFOLIO_VERSION}: tune {remaining} local learning-rate, regularisation, loss-weight, sampling and schedule policies around structural lineage {parent_id[:12]} without changing its architecture."
            ),
            evidence_references=_tree_references(
                stage="local-optuna",
                action=SearchType.OPTUNA,
                parent=parent_id,
                root=parent.root_trajectory,
            ),
        )

    local = tuple(_unique_tree_stage(local_candidates, "local-optuna").values())
    wildcard_candidates = tuple(
        item
        for item in candidates
        if item.stage == "wildcard" and item.action == SearchType.DIRECT
    )
    wildcard_identities = {
        item.evidence.trajectory_identity for item in wildcard_candidates
    }
    existing_25 = {
        item.evidence.trajectory_identity
        for item in (*root_items, *structural, *local, *wildcard_candidates)
    }
    if len(wildcard_identities) < policy.wildcard_count:
        wildcard_parents = _tree_cohort(
            (*structural, *local),
            target=policy.wildcard_count,
            wildcard_count=1,
        )
        parent = wildcard_parents[len(wildcard_identities)]
        configuration = _v7_controlled_wildcard(parent.evidence, existing_25)
        return _request(
            original,
            run_id=run_id,
            cycle=cycle,
            stage="v7-controlled-wildcard",
            search_type=SearchType.DIRECT,
            search_space=configuration,
            experiment_budget=1,
            rationale=(
                f"{V7_MECHANISM_PORTFOLIO_VERSION}: execute one controlled mechanism or objective escape from lineage {parent.evidence.trajectory_identity[:12]}."
            ),
            evidence_references=_tree_references(
                stage="wildcard",
                action=SearchType.DIRECT,
                parent=parent.evidence.trajectory_identity,
                root=parent.root_trajectory,
            ),
        )

    source = tuple((*root_items, *structural, *local, *wildcard_candidates))
    source_fidelity = 25
    for target_fidelity in (50, 100, 150):
        cohort = _tree_cohort(
            source,
            target=policy.promotion_targets[target_fidelity],
            wildcard_count=policy.wildcard_counts[target_fidelity],
        )
        completed = _unique_tree_stage(candidates, f"promote-{target_fidelity}")
        pending = next(
            (
                item
                for item in cohort
                if item.evidence.trajectory_identity not in completed
            ),
            None,
        )
        if pending is not None:
            configuration = dict(pending.evidence.configuration)
            configuration["maximum_epochs"] = target_fidelity
            return _request(
                original,
                run_id=run_id,
                cycle=cycle,
                stage=f"v7-promote-{source_fidelity}-{target_fidelity}",
                search_type=SearchType.DIRECT,
                search_space=configuration,
                experiment_budget=1,
                rationale=(
                    f"{V7_MECHANISM_PORTFOLIO_VERSION}: continue diverse lineage {pending.evidence.trajectory_identity[:12]} from {source_fidelity} to {target_fidelity} epochs."
                ),
                evidence_references=_tree_references(
                    stage=f"promote-{target_fidelity}",
                    action=pending.action,
                    parent=pending.evidence.trajectory_identity,
                    root=pending.root_trajectory,
                    extra=(
                        pending.evidence.experiment_id,
                        f"promotion-from-epoch:{source_fidelity}",
                        f"origin-search-type:{pending.action.value}",
                    ),
                ),
            )
        source = tuple(completed.values())
        source_fidelity = target_fidelity
    return None


def apply_v7_deadline_graduation_policy(
    original: SearchRequest,
    *,
    run_id: str,
    cycle: int,
    events: tuple[DecisionEvent, ...],
    runtime_context: TaskRuntimeContext,
) -> SearchRequest | None:
    """Stop exploration and graduate the strongest diverse pair to epoch 150."""

    policy = V7MechanismPortfolioPolicy.from_runtime(runtime_context)
    candidates = _tree_candidates(events, _evidence(events))
    highest: dict[str, TreeCandidate] = {}
    for item in candidates:
        identity = item.evidence.trajectory_identity
        current = highest.get(identity)
        if current is None or item.evidence.fidelity > current.evidence.fidelity:
            highest[identity] = item
    if not highest:
        return None
    completed = tuple(
        item for item in highest.values() if item.evidence.fidelity == 150
    )
    remaining_slots = policy.promotion_targets[150] - len(completed)
    if remaining_slots <= 0:
        return None
    pool = tuple(item for item in highest.values() if item.evidence.fidelity < 150)
    if not pool:
        return None
    represented_roots = {item.root_trajectory for item in completed}
    diverse_pool = tuple(
        item for item in pool if item.root_trajectory not in represented_roots
    )
    finalists = _tree_cohort(
        diverse_pool if diverse_pool else pool,
        target=min(remaining_slots, len(pool)),
        wildcard_count=0,
    )
    pending = finalists[0] if finalists else None
    if pending is None:
        return None
    configuration = dict(pending.evidence.configuration)
    configuration["maximum_epochs"] = 150
    return _request(
        original,
        run_id=run_id,
        cycle=cycle,
        stage="v7-deadline-graduation",
        search_type=SearchType.DIRECT,
        search_space=configuration,
        experiment_budget=1,
        rationale=(
            f"{V7_MECHANISM_PORTFOLIO_VERSION}: protected deadline mode; stop new exploration and continue diverse finalist {pending.evidence.trajectory_identity[:12]} from {pending.evidence.fidelity} directly to 150 epochs."
        ),
        evidence_references=_tree_references(
            stage="promote-150",
            action=pending.action,
            parent=pending.evidence.trajectory_identity,
            root=pending.root_trajectory,
            extra=(
                pending.evidence.experiment_id,
                f"promotion-from-epoch:{pending.evidence.fidelity}",
                f"origin-search-type:{pending.action.value}",
                "graduation-mode:protected-deadline",
            ),
        ),
    )


def apply_portfolio_policy(
    original: SearchRequest,
    *,
    run_id: str,
    cycle: int,
    events: tuple[DecisionEvent, ...],
    runtime_context: TaskRuntimeContext,
) -> SearchRequest | None:
    raw_policy = runtime_context.task_options.get("campaign_portfolio")
    if (
        isinstance(raw_policy, dict)
        and raw_policy.get("version") == V7_MECHANISM_PORTFOLIO_VERSION
    ):
        return apply_v7_mechanism_portfolio_policy(
            original,
            run_id=run_id,
            cycle=cycle,
            events=events,
            runtime_context=runtime_context,
        )
    if isinstance(raw_policy, dict) and raw_policy.get("version") in {
        TREE_PORTFOLIO_VERSION,
        V6_TREE_PORTFOLIO_VERSION,
    }:
        return apply_tree_portfolio_policy(
            original,
            run_id=run_id,
            cycle=cycle,
            events=events,
            runtime_context=runtime_context,
        )
    policy = PortfolioPolicy.from_runtime(runtime_context)
    if policy is None:
        return original
    rows = _evidence(events)
    rung25 = _unique_at_fidelity(rows, 25)

    optuna_count = len(
        {
            row.trajectory_identity
            for row in rows
            if row.fidelity == 25 and row.search_type == SearchType.OPTUNA
        }
    )
    if optuna_count < policy.screening[SearchType.OPTUNA]:
        remaining = policy.screening[SearchType.OPTUNA] - optuna_count
        return _request(
            original,
            run_id=run_id,
            cycle=cycle,
            stage="screen-optuna",
            search_type=SearchType.OPTUNA,
            search_space={"fixed": {"maximum_epochs": 25}},
            experiment_budget=remaining,
            rationale=(
                f"{PORTFOLIO_VERSION}: complete the mandatory 36-candidate native Optuna screening tranche at 25 epochs."
            ),
        )

    non_oe = {
        row.trajectory_identity
        for row in rows
        if row.fidelity == 25 and row.search_type != SearchType.OPENEVOLVE
    }
    novel_oe = {
        row.trajectory_identity
        for row in rows
        if row.fidelity == 25
        and row.search_type == SearchType.OPENEVOLVE
        and row.trajectory_identity not in non_oe
    }
    if len(novel_oe) < policy.screening[SearchType.OPENEVOLVE]:
        remaining = policy.screening[SearchType.OPENEVOLVE] - len(novel_oe)
        evaluations = remaining + 1  # imported incumbent seed plus novel mutations
        ranked_seeds = sorted(
            rung25.values(),
            key=lambda item: (item.rung_score, item.trajectory_identity),
            reverse=True,
        )
        if not ranked_seeds:
            raise ValueError("feta_unet_campaign_openevolve_seed_missing")
        incumbent = ranked_seeds[0]
        search_space = default_openevolve_configuration(
            candidate_evaluations=evaluations
        )
        search_space["openevolve"].update(
            {
                "maximum_failed_candidates": evaluations,
                "maximum_consecutive_failures": evaluations,
            }
        )
        search_space["campaign_context"] = {
            "incumbent_training_policy": policy_from_configuration(
                incumbent.configuration
            ).model_dump(mode="json"),
            "incumbent_primary_score": incumbent.rung_score,
            "incumbent_search_type": incumbent.search_type.value,
            "incumbent_experiment_id": incumbent.experiment_id,
            "prior_verified_results": [
                {
                    "search_type": item.search_type.value,
                    "primary_score": item.rung_score,
                    "configuration": item.configuration,
                }
                for item in ranked_seeds[:12]
            ],
        }
        return _request(
            original,
            run_id=run_id,
            cycle=cycle,
            stage="screen-openevolve",
            search_type=SearchType.OPENEVOLVE,
            search_space=search_space,
            experiment_budget=evaluations,
            rationale=(
                f"{PORTFOLIO_VERSION}: evolve {remaining} novel 25-epoch policies from the strongest verified Optuna seed."
            ),
        )

    direct_novel = {
        row.trajectory_identity
        for row in rows
        if row.fidelity == 25 and row.search_type == SearchType.DIRECT
    }
    if len(direct_novel) < policy.screening[SearchType.DIRECT]:
        existing = set(rung25)
        candidate = next(
            (
                item
                for item in policy.direct_screening_configurations
                if trajectory_identity(FeTAUNetSearchConfiguration.model_validate(item))
                not in existing
            ),
            None,
        )
        if candidate is None:
            raise ValueError("feta_unet_campaign_direct_screening_pool_exhausted")
        return _request(
            original,
            run_id=run_id,
            cycle=cycle,
            stage="screen-direct",
            search_type=SearchType.DIRECT,
            search_space=candidate,
            experiment_budget=1,
            rationale=(
                f"{PORTFOLIO_VERSION}: execute one deduplicated targeted DIRECT ablation at the 25-epoch screening rung."
            ),
        )

    source_fidelity = 25
    for target_fidelity in (50, 100, 150):
        target_count = policy.promotion_targets[target_fidelity]
        source = _unique_at_fidelity(rows, source_fidelity)
        completed = _unique_at_fidelity(rows, target_fidelity)
        cohort = _promotion_cohort(
            source,
            target=target_count,
            wildcard_count=policy.wildcard_counts[target_fidelity],
        )
        pending = next(
            (item for item in cohort if item.trajectory_identity not in completed),
            None,
        )
        if pending is not None:
            configuration = dict(pending.configuration)
            configuration["maximum_epochs"] = target_fidelity
            return _request(
                original,
                run_id=run_id,
                cycle=cycle,
                stage=f"promote-{source_fidelity}-{target_fidelity}",
                search_type=SearchType.DIRECT,
                search_space=configuration,
                experiment_budget=1,
                rationale=(
                    f"{PORTFOLIO_VERSION}: promote one {pending.search_type.value} lineage from {source_fidelity} to {target_fidelity} epochs using equal-rung evidence."
                ),
                evidence_references=(
                    pending.experiment_id,
                    f"promotion-from-epoch:{source_fidelity}",
                    f"origin-search-type:{pending.search_type.value}",
                ),
            )
        source_fidelity = target_fidelity
    return None
