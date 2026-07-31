MATCH (g:Gene)
WHERE coalesce(g.curie, g.hgnc_id) IN $gene_curies
OPTIONAL MATCH (s:Signature)-[membership:INCLUDES]->(g)
OPTIONAL MATCH (g)-[participation:PARTICIPATES_IN]->(p:Pathway)
OPTIONAL MATCH (p)-[:PART_OF]->(parent:Pathway)
WITH g, s, membership, p, participation, parent
WHERE s IS NOT NULL OR p IS NOT NULL
WITH
  g,
  coalesce(s, p) AS target,
  CASE WHEN s IS NOT NULL THEN 'INCLUDES' ELSE 'PARTICIPATES_IN' END AS predicate,
  coalesce(membership, participation) AS evidence,
  parent
RETURN
  {
    source_id: 'source:' + coalesce(evidence.source, target.source, 'unknown') + ':' + coalesce(target.version, 'configured-content-version'),
    source_type: 'ONTOLOGY_RELEASE',
    title: coalesce(evidence.source, target.source, 'registered pathway/signature source'),
    version: coalesce(target.version, 'configured-content-version'),
    publisher_or_database: coalesce(evidence.source, target.source, 'registered graph source'),
    asserted_by: 'curator',
    accession: coalesce(target.curie, target.sig_id)
  } AS source,
  [
    {
      curie: coalesce(g.curie, g.hgnc_id),
      entity_type: 'Gene',
      name: coalesce(g.name, g.symbol),
      safe_properties: {symbol: g.symbol}
    },
    {
      curie: CASE WHEN s IS NOT NULL THEN 'MSIGDB:' + target.sig_id ELSE target.curie END,
      entity_type: CASE WHEN s IS NOT NULL THEN 'Signature' ELSE 'Pathway' END,
      name: target.name,
      safe_properties: {parent_curie: parent.curie}
    }
  ] AS entities,
  {
    subject_curie: coalesce(g.curie, g.hgnc_id),
    predicate: predicate,
    object_curie: CASE WHEN s IS NOT NULL THEN 'MSIGDB:' + target.sig_id ELSE target.curie END,
    method: 'registered ontology or curated membership',
    confidence: 1.0,
    asserted_by: 'curator',
    trust_tier: 'CURATED',
    safe_properties: {}
  } AS assertion,
  {
    reference_type: 'GENE_CONTEXT',
    concise_claim: 'The gene has a registered signature membership or pathway participation.',
    relevant_parameters: ['alpha', 'K']
  } AS reference
ORDER BY g.symbol, predicate, target.name
LIMIT $limit
