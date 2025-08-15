STRICT_RAG_JSON_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "rag_summary",
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["bullets", "evidence", "confidence"],
            "properties": {
                "bullets": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                    "maxItems": 6
                },
                "evidence": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["doc_id", "span"],
                        "properties": {
                            "doc_id": {"type": "integer"},     # <-- integer, not string
                            "span":   {"type": "string", "maxLength": 400}
                        }
                    },
                    "minItems": 1,                           # <-- at least one item
                    "maxItems": 12
                },
                "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0}
            }
        }
    }
}
