MATCH (n:Network)
WHERE n.codename IN $codenames
WITH n, coalesce(n.id, 'network:' + toLower(n.codename)) AS curie
RETURN
  {
    source_id: 'source:network-catalog:' + coalesce(n.source, 'unknown'),
    source_type: 'CURATED_DATABASE',
    title: coalesce(n.source, 'iCCA network catalog'),
    version: coalesce(n.version, 'configured-content-version'),
    publisher_or_database: coalesce(n.source, 'iCCA network catalog'),
    asserted_by: 'curator',
    accession: curie
  } AS source,
  [{
    curie: curie,
    entity_type: 'Network',
    name: coalesce(n.name, n.codename),
    safe_properties: {
      codename: n.codename,
      network_type: n.type,
      description: n.description,
      metadata_only: coalesce(n.metadata_only, true)
    }
  }] AS entities,
  {
    subject_curie: curie,
    predicate: 'CATALOGUED_AS',
    object_curie: curie,
    method: 'task-compatible network metadata catalog',
    confidence: 1.0,
    asserted_by: 'curator',
    trust_tier: 'CURATED',
    safe_properties: {codename: n.codename}
  } AS assertion,
  {
    reference_type: 'NETWORK_CATALOG',
    concise_claim: 'The network is a registered metadata-only iCCA network choice.',
    relevant_parameters: ['network']
  } AS reference
ORDER BY n.codename, curie
LIMIT $limit
