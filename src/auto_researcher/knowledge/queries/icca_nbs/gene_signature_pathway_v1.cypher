MATCH (g:Gene)
WHERE coalesce(g.curie, g.hgnc_id) IN $gene_curies
OPTIONAL MATCH (s:Signature)-[membership:INCLUDES]->(g)
WITH
  g,
  collect(
    CASE
      WHEN s IS NULL THEN null
      ELSE {
        target: s,
        predicate: 'INCLUDES',
        evidence: membership,
        parent: null,
        is_signature: true
      }
    END
  ) AS signature_contexts
OPTIONAL MATCH (g)-[participation:PARTICIPATES_IN]->(p:Pathway)
OPTIONAL MATCH (p)-[:PART_OF]->(parent:Pathway)
WITH
  g,
  signature_contexts,
  collect(
    CASE
      WHEN p IS NULL THEN null
      ELSE {
        target: p,
        predicate: 'PARTICIPATES_IN',
        evidence: participation,
        parent: parent,
        is_signature: false
      }
    END
  ) AS pathway_contexts
UNWIND signature_contexts + pathway_contexts AS context
WITH
  g,
  context.target AS target,
  context.predicate AS predicate,
  context.evidence AS evidence,
  context.parent AS parent,
  context.is_signature AS is_signature,
  coalesce(g.curie, g.hgnc_id) AS gene_curie,
  CASE
    WHEN NOT context.is_signature THEN context.target.curie
    WHEN coalesce(context.target.curie, context.target.sig_id) CONTAINS ':'
      THEN coalesce(context.target.curie, context.target.sig_id)
    ELSE 'MSIGDB:' + coalesce(context.target.curie, context.target.sig_id)
  END AS target_curie
RETURN
  {
    source_id: 'source:' + coalesce(evidence.source, target.source, 'unknown') + ':' + coalesce(target.version, 'configured-content-version'),
    source_type: 'ONTOLOGY_RELEASE',
    title: coalesce(evidence.source, target.source, 'registered pathway/signature source'),
    version: coalesce(target.version, 'configured-content-version'),
    publisher_or_database: coalesce(evidence.source, target.source, 'registered graph source'),
    asserted_by: 'curator',
    accession: target_curie
  } AS source,
  [
    {
      curie: gene_curie,
      entity_type: 'Gene',
      name: coalesce(g.name, g.symbol),
      safe_properties: {symbol: g.symbol}
    },
    {
      curie: target_curie,
      entity_type: CASE WHEN is_signature THEN 'Signature' ELSE 'Pathway' END,
      name: target.name,
      safe_properties: {parent_curie: parent.curie}
    }
  ] AS entities,
  {
    subject_curie: CASE WHEN is_signature THEN target_curie ELSE gene_curie END,
    predicate: predicate,
    object_curie: CASE WHEN is_signature THEN gene_curie ELSE target_curie END,
    method: 'registered ontology or curated membership',
    confidence: 1.0,
    asserted_by: 'curator',
    trust_tier: 'CURATED',
    safe_properties: {}
  } AS assertion,
  {
    reference_type: 'GENE_CONTEXT',
    concise_claim: 'The configured gene is included by a registered signature or participates in a registered pathway.',
    relevant_parameters: ['alpha', 'K']
  } AS reference
ORDER BY g.symbol, predicate, target.name
LIMIT $limit
