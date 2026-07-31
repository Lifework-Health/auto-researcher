MATCH (n)
WHERE (n:Gene OR n:Pathway OR n:Disease OR n:Signature OR n:Network)
  AND (
    coalesce(n.curie, n.hgnc_id, n.id) = $curie
    OR (n:Signature AND 'MSIGDB:' + n.curie = $curie)
  )
WITH n,
     CASE
       WHEN n:Signature AND NOT n.curie CONTAINS ':' THEN 'MSIGDB:' + n.curie
       ELSE coalesce(n.curie, n.hgnc_id, n.id)
     END AS curie
RETURN
  {
    source_id: 'source:' + coalesce(n.source, 'unknown') + ':' + coalesce(n.version, 'unversioned'),
    source_type: 'CURATED_DATABASE',
    title: coalesce(n.source, 'registered graph source'),
    version: coalesce(n.version, 'configured-content-version'),
    publisher_or_database: coalesce(n.source, 'registered graph source'),
    asserted_by: 'curator',
    accession: curie
  } AS source,
  [{
    curie: curie,
    entity_type: head(labels(n)),
    name: coalesce(n.name, n.symbol, n.codename, curie),
    safe_properties: {}
  }] AS entities,
  {
    subject_curie: curie,
    predicate: 'IDENTIFIES',
    object_curie: curie,
    method: 'exact stable identifier lookup',
    confidence: 1.0,
    asserted_by: 'curator',
    trust_tier: 'CURATED',
    safe_properties: {}
  } AS assertion,
  {
    reference_type: 'ENTITY_IDENTITY',
    concise_claim: 'The registered graph resolves the supplied stable identifier.',
    relevant_parameters: []
  } AS reference
ORDER BY curie
LIMIT $limit
