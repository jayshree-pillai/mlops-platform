# Central registry for JSON schemas used in prompt outputs.

STRICT_RAG_JSON_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "rag_summary",
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["bullets", "confidence"],
            "properties": {
                "bullets": {
                    "type": "array",
                    "items": {"type": "string"},
                    "maxItems": 6
                },
                "confidence": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0
                },
                "evidence": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["quote"],
                        "properties": {
                            "quote": {"type": "string", "maxLength": 400}
                        }
                    },
                    "maxItems": 6
                }
            }
        }
    }
}
