CALL db.labels() YIELD label
WITH label
ORDER BY label
WITH collect(label) AS nested_labels
CALL db.relationshipTypes() YIELD relationshipType
WITH nested_labels, relationshipType
ORDER BY relationshipType
WITH nested_labels, collect(relationshipType) AS relationships
RETURN nested_labels AS labels,
       relationships
ORDER BY labels
LIMIT $limit
