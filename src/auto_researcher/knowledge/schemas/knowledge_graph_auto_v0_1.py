"""Compatibility profile for knowledge_graph_auto at its inspected main commit."""

from dataclasses import dataclass

KNOWLEDGE_GRAPH_AUTO_COMMIT = "759090a220148fbe360f4fc519561fa41cb0bfdc"


@dataclass(frozen=True)
class SchemaPreflightResult:
    passed: bool
    missing_labels: tuple[str, ...]
    missing_relationships: tuple[str, ...]
    curie_coverage: dict[str, tuple[int, int]]
    graph_counts: dict[str, int]
    warnings: tuple[str, ...] = ()


class KnowledgeGraphAutoProfile:
    profile_id = "knowledge-graph-auto-v0.1"
    loaded_backbone_labels = frozenset(
        {"Gene", "CellState", "Signature", "Pathway", "Network", "Disease"}
    )
    cohort_gated_labels = frozenset({"Subtype", "ClinicalCovariate", "Cohort"})
    loaded_relationships = frozenset(
        {
            "IS_A",
            "REGULATES",
            "INCLUDES",
            "PARTICIPATES_IN",
            "PART_OF",
            "EXPRESSES_MARKER",
            "HAS_SIGNATURE",
            "DIFFERENTIATES_TO",
        }
    )
    cohort_gated_relationships = frozenset(
        {
            "SUBTYPE_OF",
            "IMPLICATED_IN",
            "DEFINED_BY",
            "HAS_SURVIVAL_PROFILE",
            "PROGNOSTIC_IN",
            "HAS_IMMUNE_PHENOTYPE",
            "PROFILES",
        }
    )
    curie_properties = {
        "Gene": "hgnc_id",
        "CellState": "cl_id",
        "Signature": "curie",
        "Pathway": "curie",
        "Disease": "curie",
        "Network": "id",
    }

    def preflight(
        self,
        *,
        labels: set[str],
        relationships: set[str],
        curie_coverage: dict[str, tuple[int, int]],
        graph_counts: dict[str, int],
        required_labels: set[str],
        required_relationships: set[str],
    ) -> SchemaPreflightResult:
        missing_labels = tuple(sorted(required_labels - labels))
        missing_relationships = tuple(sorted(required_relationships - relationships))
        warnings = []
        for label in sorted(required_labels):
            covered, total = curie_coverage.get(label, (0, 0))
            if total and covered < total:
                warnings.append(
                    f"{label} stable identifier coverage is {covered}/{total}"
                )
        if "MEMBER_OF" in relationships and "INCLUDES" not in relationships:
            warnings.append(
                "Schema uses MEMBER_OF, but inspected graph profile requires INCLUDES."
            )
        return SchemaPreflightResult(
            passed=not missing_labels and not missing_relationships,
            missing_labels=missing_labels,
            missing_relationships=missing_relationships,
            curie_coverage=curie_coverage,
            graph_counts=graph_counts,
            warnings=tuple(warnings),
        )
