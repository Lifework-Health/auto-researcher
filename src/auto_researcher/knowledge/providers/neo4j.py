"""Evidence-safe Neo4j provider using fixed registered read templates."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
import time
from typing import Any

from auto_researcher.knowledge.identity import (
    assertion_id,
    bundle_content_hash,
    content_hash,
    entity_id,
    reference_id,
    stable_identifier,
)
from auto_researcher.knowledge.models import (
    KnowledgeAssertion,
    KnowledgeBundle,
    KnowledgeEntity,
    KnowledgeErrorCode,
    KnowledgeGraphSnapshot,
    KnowledgeProviderConfiguration,
    KnowledgeReadinessCheck,
    KnowledgeReadinessResult,
    KnowledgeReference,
    KnowledgeRetrievalRequest,
    KnowledgeSource,
    KnowledgeValidationResult,
)
from auto_researcher.knowledge.protocols import KnowledgeProviderError
from auto_researcher.knowledge.templates import KnowledgeQueryTemplateRegistry


class Neo4jKnowledgeProvider:
    provider_id = "neo4j"
    provider_version = "6.2.0-adapter-v1"

    def __init__(
        self,
        configuration: KnowledgeProviderConfiguration,
        template_registry: KnowledgeQueryTemplateRegistry,
        *,
        uri: str | None = None,
        username: str | None = None,
        password: str | None = None,
        driver: Any | None = None,
        clock: Callable[[], datetime],
        query_factory: Callable[[str, float], Any] | None = None,
        schema_profile: Any | None = None,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.configuration = configuration
        self._templates = template_registry
        self._clock = clock
        self._closed = False
        self._query_factory = query_factory
        self._schema_profile = schema_profile
        self._monotonic = monotonic
        self._driver = driver
        if driver is None and uri and username and password:
            try:
                from neo4j import GraphDatabase
            except ImportError as exc:
                raise KnowledgeProviderError(
                    KnowledgeErrorCode.PROVIDER_NOT_INSTALLED.value
                ) from exc
            try:
                self._driver = GraphDatabase.driver(
                    uri,
                    auth=(username, password),
                )
            except Exception:
                raise KnowledgeProviderError(
                    KnowledgeErrorCode.PROVIDER_NOT_CONFIGURED.value
                ) from None

    def _query(self, text: str, timeout_seconds: float) -> Any:
        if self._query_factory is not None:
            return self._query_factory(text, timeout_seconds)
        return text

    def execution_template_hashes(self) -> dict[str, str]:
        template = self._templates.get("generic.schema_preflight", "1.0.0")
        return {f"{template.template_id}@{template.version}": template.cypher_sha256}

    def readiness(
        self,
        configuration: KnowledgeProviderConfiguration,
    ) -> KnowledgeReadinessResult:
        checks: list[KnowledgeReadinessCheck] = []
        errors: list[KnowledgeErrorCode] = []
        warnings: list[str] = []
        identity_matches = (
            configuration.provider_id == self.provider_id
            and configuration == self.configuration
        )
        checks.append(
            KnowledgeReadinessCheck(
                code="provider_identity",
                passed=identity_matches,
                message=(
                    "Neo4j provider identity matches configuration."
                    if identity_matches
                    else "Neo4j provider configuration identity does not match."
                ),
            )
        )
        if not identity_matches:
            errors.append(KnowledgeErrorCode.PROVIDER_NOT_CONFIGURED)
        configured = (
            self._driver is not None and configuration.enabled and identity_matches
        )
        checks.append(
            KnowledgeReadinessCheck(
                code="provider_configured",
                passed=configured,
                message=(
                    "Neo4j driver is configured."
                    if configured
                    else "Neo4j provider is disabled or lacks runtime credentials."
                ),
            )
        )
        if not configured:
            errors.append(KnowledgeErrorCode.PROVIDER_NOT_CONFIGURED)
        else:
            try:
                self._driver.verify_connectivity()
                checks.append(
                    KnowledgeReadinessCheck(
                        code="connectivity",
                        passed=True,
                        message="Neo4j connectivity verified.",
                    )
                )
            except Exception as exc:
                checks.append(
                    KnowledgeReadinessCheck(
                        code="connectivity",
                        passed=False,
                        message="Neo4j connectivity failed.",
                    )
                )
                errors.append(self._safe_error(exc))
        read_only_verified = False
        if configured and not errors:
            try:
                with self._driver.session(database=configuration.database) as session:
                    result = session.run(
                        "SHOW CURRENT USER PRIVILEGES "
                        "YIELD access, action "
                        "RETURN access, action ORDER BY action"
                    )
                    rows = [dict(row) for row in result]
                    summary = result.consume()
                self._assert_no_updates(summary)
                granted = {
                    str(row.get("action", "")).casefold()
                    for row in rows
                    if str(row.get("access", "")).casefold() == "granted"
                }
                read_only_verified = not any(
                    token in action
                    for action in granted
                    for token in ("write", "create", "delete", "set property")
                )
            except Exception:
                warnings.append(
                    "Read-only privilege verification is unavailable; no test write was attempted."
                )
        checks.append(
            KnowledgeReadinessCheck(
                code="read_only_privileges",
                passed=(
                    read_only_verified or not configuration.require_verified_read_only
                ),
                message=(
                    "Configured account exposes no granted write privileges."
                    if read_only_verified
                    else "Read-only privileges could not be verified."
                ),
            )
        )
        if configuration.require_verified_read_only and not read_only_verified:
            errors.append(KnowledgeErrorCode.READ_ONLY_NOT_VERIFIED)
        return KnowledgeReadinessResult(
            ready=not errors,
            checks=tuple(checks),
            errors=tuple(dict.fromkeys(errors)),
            warnings=tuple(warnings),
            provider_id=self.provider_id,
            provider_version=self.provider_version,
            schema_version=configuration.schema_version,
            content_version=configuration.content_version,
        )

    def retrieve(self, request: KnowledgeRetrievalRequest) -> KnowledgeBundle:
        if self._driver is None:
            raise KnowledgeProviderError(
                KnowledgeErrorCode.PROVIDER_NOT_CONFIGURED.value
            )
        all_rows: list[dict[str, Any]] = []
        deadline = (
            self._monotonic() + self.configuration.query_timeout_seconds
        )
        required_labels: set[str] = set()
        required_relationships: set[str] = set()
        prepared: list[tuple[Any, dict[str, Any], int]] = []
        for item in request.query_plan.template_requests:
            try:
                template, parameters = self._templates.validate_request(
                    item,
                    task_id=request.task_id,
                    schema_version=request.schema_version,
                )
            except KeyError as exc:
                raise KnowledgeProviderError(
                    KnowledgeErrorCode.UNKNOWN_QUERY_TEMPLATE.value
                ) from exc
            except ValueError as exc:
                raise KnowledgeProviderError(
                    KnowledgeErrorCode.INVALID_QUERY_PARAMETERS.value
                ) from exc
            key = f"{template.template_id}@{template.version}"
            if request.template_hashes.get(key) != template.cypher_sha256:
                raise KnowledgeProviderError(
                    KnowledgeErrorCode.UNKNOWN_QUERY_TEMPLATE.value
                )
            required_labels.update(template.allowed_labels)
            required_relationships.update(template.allowed_relationships)
            prepared.append((template, parameters, item.maximum_records))
        preflight_metadata = self._schema_preflight(
            request,
            required_labels=required_labels,
            required_relationships=required_relationships,
            deadline=deadline,
        )
        for template, parameters, maximum_records in prepared:
            try:
                rows = self._execute_read(
                    template.cypher,
                    parameters,
                    deadline=deadline,
                )
            except KnowledgeProviderError:
                raise
            except Exception as exc:
                raise KnowledgeProviderError(self._safe_error(exc).value) from None
            if len(rows) > maximum_records:
                raise KnowledgeProviderError(
                    KnowledgeErrorCode.RESULT_LIMIT_EXCEEDED.value
                )
            all_rows.extend(rows)
            if len(all_rows) > request.query_plan.maximum_total_records:
                raise KnowledgeProviderError(
                    KnowledgeErrorCode.RESULT_LIMIT_EXCEEDED.value
                )
        try:
            return self._bundle_from_rows(
                request,
                all_rows,
                required_labels=required_labels,
                required_relationships=required_relationships,
                preflight_metadata=preflight_metadata,
            )
        except KnowledgeProviderError:
            raise
        except (KeyError, TypeError, ValueError):
            raise KnowledgeProviderError(
                KnowledgeErrorCode.INVALID_PROVENANCE.value
            ) from None

    def _schema_preflight(
        self,
        request: KnowledgeRetrievalRequest,
        *,
        required_labels: set[str],
        required_relationships: set[str],
        deadline: float,
    ) -> dict[str, Any]:
        template = self._templates.get("generic.schema_preflight", "1.0.0")
        key = f"{template.template_id}@{template.version}"
        if request.template_hashes.get(key) != template.cypher_sha256:
            raise KnowledgeProviderError(
                KnowledgeErrorCode.UNKNOWN_QUERY_TEMPLATE.value
            )
        try:
            rows = self._execute_read(
                template.cypher,
                {"limit": 1},
                deadline=deadline,
            )
        except KnowledgeProviderError:
            raise
        except Exception as exc:
            raise KnowledgeProviderError(self._safe_error(exc).value) from None
        if len(rows) != 1:
            raise KnowledgeProviderError(KnowledgeErrorCode.SCHEMA_MISMATCH.value)
        labels = {str(item) for item in rows[0].get("labels", ())}
        relationships = {str(item) for item in rows[0].get("relationships", ())}
        profile_passed = True
        if self._schema_profile is not None:
            profile = self._schema_profile.preflight(
                labels=labels,
                relationships=relationships,
                curie_coverage={},
                graph_counts={},
                required_labels=required_labels,
                required_relationships=required_relationships,
            )
            profile_passed = profile.passed
        if (
            not profile_passed
            or not required_labels.issubset(labels)
            or not required_relationships.issubset(relationships)
        ):
            raise KnowledgeProviderError(KnowledgeErrorCode.SCHEMA_MISMATCH.value)
        return {
            "preflight_passed": True,
            "observed_label_count": len(labels),
            "observed_relationship_type_count": len(relationships),
        }

    def _execute_read(
        self,
        cypher: str,
        parameters: dict[str, Any],
        *,
        deadline: float,
    ) -> list[dict[str, Any]]:
        for attempt in range(1, self.configuration.maximum_attempts + 1):
            remaining = deadline - self._monotonic()
            if remaining <= 0:
                raise KnowledgeProviderError(
                    KnowledgeErrorCode.QUERY_TIMEOUT.value
                )

            def work(transaction):
                result = transaction.run(
                    self._query(cypher, remaining),
                    parameters,
                )
                rows = [_safe_plain(dict(record)) for record in result]
                summary = result.consume()
                self._assert_no_updates(summary)
                return rows

            managed_work = work
            if self._query_factory is None:
                try:
                    from neo4j import unit_of_work
                except ImportError:
                    pass
                else:
                    managed_work = unit_of_work(timeout=remaining)(work)
            try:
                with self._driver.session(
                    database=self.configuration.database
                ) as session:
                    return session.execute_read(managed_work)
            except KnowledgeProviderError:
                raise
            except Exception as exc:
                code = self._safe_error(exc)
                if (
                    code == KnowledgeErrorCode.TRANSIENT_QUERY_FAILURE
                    and attempt < self.configuration.maximum_attempts
                ):
                    continue
                raise KnowledgeProviderError(code.value) from None
        raise KnowledgeProviderError(KnowledgeErrorCode.TRANSIENT_QUERY_FAILURE.value)

    def _assert_no_updates(self, summary: Any) -> None:
        counters = getattr(summary, "counters", None)
        if counters and (
            bool(getattr(counters, "contains_updates", False))
            or bool(getattr(counters, "contains_system_updates", False))
        ):
            raise KnowledgeProviderError(
                KnowledgeErrorCode.FORBIDDEN_WRITE_DETECTED.value
            )

    def _bundle_from_rows(
        self,
        request: KnowledgeRetrievalRequest,
        rows: list[dict[str, Any]],
        *,
        required_labels: set[str],
        required_relationships: set[str],
        preflight_metadata: dict[str, Any],
    ) -> KnowledgeBundle:
        bundle_id = stable_identifier("knowledge-bundle", request.retrieval_id)
        sources: dict[str, KnowledgeSource] = {}
        entities: dict[str, KnowledgeEntity] = {}
        assertions: dict[str, KnowledgeAssertion] = {}
        references: dict[str, KnowledgeReference] = {}
        for row in rows:
            source_payload = dict(row.get("source") or {})
            if not source_payload:
                continue
            if source_payload.get("version") == "configured-content-version":
                source_payload["version"] = request.content_version
            source_payload["retrieval_reference"] = request.retrieval_id
            source = KnowledgeSource.model_validate(source_payload)
            _deduplicate(sources, source.source_id, source)
            row_entities = []
            for payload in row.get("entities") or ():
                item = dict(payload)
                curie = _required_string(item.pop("curie", None), "entity curie")
                kind = _required_string(
                    item.pop("entity_type", None),
                    "entity type",
                )
                entity = KnowledgeEntity(
                    entity_id=entity_id(curie, kind),
                    curie=curie,
                    entity_type=kind,
                    name=_required_string(item.pop("name", None), "entity name"),
                    aliases=tuple(item.pop("aliases", ())),
                    safe_properties=item.pop("safe_properties", {}),
                    source_references=(source.source_id,),
                )
                _merge_entity(entities, entity)
                row_entities.append(entity)
            assertion_payload = dict(row.get("assertion") or {})
            reference_payload = dict(row.get("reference") or {})
            if len(row_entities) < 1 or not assertion_payload:
                continue
            subject_curie = _required_string(
                assertion_payload.pop("subject_curie", None),
                "assertion subject",
            )
            object_curie = _required_string(
                assertion_payload.pop("object_curie", None),
                "assertion object",
            )
            try:
                subject = next(
                    item for item in row_entities if item.curie == subject_curie
                )
                object_ = next(
                    item for item in row_entities if item.curie == object_curie
                )
            except StopIteration as exc:
                raise KnowledgeProviderError(
                    KnowledgeErrorCode.INVALID_IDENTIFIER.value
                ) from exc
            predicate = _required_string(
                assertion_payload.pop("predicate", None),
                "assertion predicate",
            )
            assertion_identifier = assertion_id(
                subject_curie,
                predicate,
                object_curie,
                (source.source_id,),
            )
            assertion = KnowledgeAssertion(
                assertion_id=assertion_identifier,
                subject_entity_id=subject.entity_id,
                predicate=predicate,
                object_entity_id=object_.entity_id,
                direction="FORWARD",
                source_references=(source.source_id,),
                **assertion_payload,
            )
            _deduplicate(assertions, assertion.assertion_id, assertion)
            reference = KnowledgeReference(
                reference_id=reference_id(assertion_identifier, bundle_id),
                reference_type=_required_string(
                    reference_payload.get("reference_type"),
                    "reference type",
                ),
                concise_claim=_required_string(
                    reference_payload.get("concise_claim"),
                    "reference claim",
                ),
                subject_curie=subject_curie,
                predicate=predicate,
                object_curie=object_curie,
                source_references=(source.source_id,),
                trust_tier=assertion.trust_tier,
                confidence=assertion.confidence,
                citation_label=f"[{source.title}]",
                bundle_id=bundle_id,
                relevant_parameters=tuple(
                    reference_payload.get("relevant_parameters") or ()
                ),
            )
            _deduplicate(references, reference.reference_id, reference)
        safe_content = {
            "rows": rows,
            "required_labels": sorted(required_labels),
            "required_relationships": sorted(required_relationships),
        }
        bundle = KnowledgeBundle(
            bundle_id=bundle_id,
            retrieval_id=request.retrieval_id,
            query_plan_hash=request.query_plan_hash,
            graph_snapshot=KnowledgeGraphSnapshot(
                provider_id=self.provider_id,
                graph_alias=request.graph_alias,
                schema_version=request.schema_version,
                configured_content_version=request.content_version,
                retrieval_timestamp=self._clock(),
                safe_graph_metadata={
                    "returned_records": len(rows),
                    "required_labels": sorted(required_labels),
                    "required_relationships": sorted(required_relationships),
                    **preflight_metadata,
                },
                returned_content_hash=content_hash(safe_content),
            ),
            sources=tuple(sources[key] for key in sorted(sources)),
            entities=tuple(entities[key] for key in sorted(entities)),
            assertions=tuple(assertions[key] for key in sorted(assertions)),
            references=tuple(references[key] for key in sorted(references)),
            validation_result=KnowledgeValidationResult(
                passed=False,
                accepted_reference_count=0,
                rejected_assertion_count=0,
                reason_codes=("not_validated",),
            ),
            bundle_hash="0" * 64,
        )
        return bundle.model_copy(update={"bundle_hash": bundle_content_hash(bundle)})

    def _safe_error(self, exc: Exception) -> KnowledgeErrorCode:
        name = type(exc).__name__.casefold()
        text = str(exc).casefold()
        if "auth" in name or "unauthorized" in text:
            return KnowledgeErrorCode.AUTHENTICATION_FAILED
        if "timeout" in name or "timed out" in text:
            return KnowledgeErrorCode.QUERY_TIMEOUT
        if any(
            token in name
            for token in ("serviceunavailable", "transient", "sessionexpired")
        ):
            return KnowledgeErrorCode.TRANSIENT_QUERY_FAILURE
        return KnowledgeErrorCode.CONNECTIVITY_FAILED

    def close(self) -> None:
        if not self._closed and self._driver is not None:
            try:
                self._driver.close()
            except Exception:
                self._closed = True
                raise KnowledgeProviderError(
                    KnowledgeErrorCode.CONNECTIVITY_FAILED.value
                ) from None
        self._closed = True


def _safe_plain(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {
            str(key): _safe_plain(item)
            for key, item in value.items()
            if str(key).casefold() not in {"id", "_id", "element_id"}
        }
    if isinstance(value, (list, tuple)):
        return [_safe_plain(item) for item in value]
    if hasattr(value, "items"):
        return {
            str(key): _safe_plain(item)
            for key, item in value.items()
            if str(key).casefold() not in {"id", "_id", "element_id"}
        }
    raise KnowledgeProviderError(KnowledgeErrorCode.INVALID_IDENTIFIER.value)


def _deduplicate(target: dict, key: str, value: Any) -> None:
    existing = target.get(key)
    if existing is not None and existing != value:
        raise KnowledgeProviderError("conflicting_knowledge_content")
    target[key] = value


def _merge_entity(target: dict[str, KnowledgeEntity], value: KnowledgeEntity) -> None:
    existing = target.get(value.entity_id)
    if existing is None:
        target[value.entity_id] = value
        return
    if existing.model_copy(update={"source_references": ()}) != value.model_copy(
        update={"source_references": ()}
    ):
        raise KnowledgeProviderError("conflicting_knowledge_content")
    target[value.entity_id] = existing.model_copy(
        update={
            "source_references": tuple(
                sorted(set(existing.source_references) | set(value.source_references))
            )
        }
    )


def _required_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise KnowledgeProviderError(KnowledgeErrorCode.INVALID_PROVENANCE.value)
    return value
