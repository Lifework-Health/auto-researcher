"""Controller-owned search portfolio and staged-fidelity graduation policy."""

from __future__ import annotations

import hashlib
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
    CANDIDATE_CONFIGURATION_FIELDS,
    FeTAUNetSearchConfiguration,
)
from auto_researcher.tasks.feta_unet_search.continuation import trajectory_identity
from auto_researcher.tasks.feta_unet_search.openevolve import (
    default_openevolve_configuration,
    policy_from_configuration,
)
from auto_researcher.tasks.models import TaskRuntimeContext

PORTFOLIO_VERSION = "feta-unet-60-18-7-2-portfolio-v1"


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


def apply_portfolio_policy(
    original: SearchRequest,
    *,
    run_id: str,
    cycle: int,
    events: tuple[DecisionEvent, ...],
    runtime_context: TaskRuntimeContext,
) -> SearchRequest | None:
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
