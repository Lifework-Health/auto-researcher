MATCH (n)
WITH collect(DISTINCT labels(n)) AS nested_labels
MATCH ()-[relationship]->()
WITH nested_labels, collect(DISTINCT type(relationship)) AS relationships
UNWIND nested_labels AS label_group
UNWIND label_group AS label
RETURN collect(DISTINCT label) AS labels,
       relationships
ORDER BY labels
LIMIT $limit
