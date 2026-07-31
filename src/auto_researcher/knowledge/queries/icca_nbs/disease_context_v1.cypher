MATCH (d:Disease)
WHERE d.curie IN $disease_curies
MATCH (subject)-[r:IMPLICATED_IN|SUBTYPE_OF|DEFINED_BY|PROGNOSTIC_IN]->(d)
WITH subject, r, d
WITH subject, r, d,
     CASE
       WHEN subject:Signature
         AND NOT coalesce(subject.curie, subject.sig_id) CONTAINS ':'
         THEN 'MSIGDB:' + coalesce(subject.curie, subject.sig_id)
       ELSE coalesce(subject.curie, subject.hgnc_id, subject.sig_id, subject.cl_id)
     END AS subject_curie
RETURN
  {
    source_id: 'source:' + coalesce(r.source, r.pmid, d.source, 'unknown'),
    source_type: CASE WHEN r.pmid IS NULL THEN 'ONTOLOGY_RELEASE' ELSE 'LITERATURE' END,
    title: coalesce(r.source, d.source, 'registered disease context'),
    version: coalesce(d.version, 'configured-content-version'),
    publisher_or_database: coalesce(r.source, d.source, 'registered graph source'),
    asserted_by: coalesce(r.asserted_by, 'curator'),
    pmid: r.pmid,
    accession: d.curie
  } AS source,
  [
    {
      curie: subject_curie,
      entity_type: head(labels(subject)),
      name: coalesce(subject.name, subject.symbol),
      safe_properties: {}
    },
    {
      curie: d.curie,
      entity_type: 'Disease',
      name: d.name,
      safe_properties: {}
    }
  ] AS entities,
  {
    subject_curie: subject_curie,
    predicate: type(r),
    object_curie: d.curie,
    method: coalesce(r.method, 'registered disease ontology relation'),
    confidence: coalesce(r.confidence, 1.0),
    asserted_by: coalesce(r.asserted_by, 'curator'),
    trust_tier: CASE WHEN r.asserted_by = 'corpus' THEN 'CORPUS' WHEN r.asserted_by = 'llm' THEN 'LIVE' ELSE 'CURATED' END,
    safe_properties: {}
  } AS assertion,
  {
    reference_type: 'DISEASE_CONTEXT',
    concise_claim: 'A registered entity is related to the configured disease context.',
    relevant_parameters: ['K', 'alpha']
  } AS reference
ORDER BY d.curie, type(r), subject_curie
LIMIT $limit
