MATCH (d)
WHERE (d:Disease OR d:Subtype) AND d.curie IN $disease_curies
MATCH (d)-[r:HAS_IMMUNE_PHENOTYPE]->(target)
WHERE target:Signature OR target:CellState
WITH d, r, target,
     CASE
       WHEN target:Signature
         AND NOT coalesce(target.curie, target.sig_id) CONTAINS ':'
         THEN 'MSIGDB:' + coalesce(target.curie, target.sig_id)
       ELSE coalesce(target.curie, target.sig_id, target.cl_id)
     END AS target_curie
RETURN
  {
    source_id: 'source:' + coalesce(r.pmid, r.source, 'unknown'),
    source_type: CASE WHEN r.pmid IS NULL THEN 'LIVE_ASSERTION' ELSE 'LITERATURE' END,
    title: coalesce(r.source, 'registered immune phenotype assertion'),
    version: 'configured-content-version',
    publisher_or_database: coalesce(r.source, 'registered graph source'),
    asserted_by: coalesce(r.asserted_by, 'unverified'),
    pmid: r.pmid,
    accession: d.curie
  } AS source,
  [
    {
      curie: d.curie,
      entity_type: head(labels(d)),
      name: d.name,
      safe_properties: {}
    },
    {
      curie: target_curie,
      entity_type: head(labels(target)),
      name: target.name,
      safe_properties: {}
    }
  ] AS entities,
  {
    subject_curie: d.curie,
    predicate: 'HAS_IMMUNE_PHENOTYPE',
    object_curie: target_curie,
    method: coalesce(r.method, 'registered immune phenotype assertion'),
    confidence: coalesce(r.confidence, 0.0),
    asserted_by: coalesce(r.asserted_by, 'unverified'),
    trust_tier: CASE WHEN r.asserted_by = 'corpus' THEN 'CORPUS' WHEN r.asserted_by = 'curator' THEN 'CURATED' WHEN r.asserted_by = 'llm' THEN 'LIVE' ELSE 'UNVERIFIED' END,
    safe_properties: {}
  } AS assertion,
  {
    reference_type: 'IMMUNE_BRIDGE',
    concise_claim: 'The configured disease context has a registered immune phenotype.',
    relevant_parameters: ['K', 'alpha']
  } AS reference
ORDER BY d.curie, target_curie
LIMIT $limit
