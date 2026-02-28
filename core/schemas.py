"""
JSON Schemas for structured LLM output.

These schemas follow OpenAI's json_schema format for structured output.
Used with response_format parameter in LLM invoke calls.
"""

# Schema 1: ENTITY_RETRIEVE - returns {endpoint, entities[]}
ENTITY_RETRIEVE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "entity_retrieve",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "endpoint": {"type": "boolean"},
                "entities": {
                    "type": "array",
                    "items": {"type": "string"}
                }
            },
            "required": ["endpoint", "entities"],
            "additionalProperties": False
        }
    }
}

# Schema 2: INIT_GRAPH_ENTITY - returns {entities: [{id, name, description, weight, uncertainty, confidential_level, relevance}]}
ENTITY_NODES_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "entity_nodes",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "entities": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "name": {"type": "string"},
                            "description": {"type": "string"},
                            "weight": {"type": "number"},
                            "uncertainty": {"type": "number"},
                            "confidential_level": {"type": "string"},
                            "relevance": {"type": "number"}
                        },
                        "required": ["id", "name", "description", "weight", "uncertainty", "confidential_level", "relevance"]
                    }
                }
            },
            "required": ["entities"],
            "additionalProperties": False
        }
    }
}

# Schema 3: INIT_ENTITY_GRAPH_EDGES / INIT_RELATION_GRAPH_EDGES - returns {endpoint, edges[]}
ENTITY_EDGES_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "entity_edges",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "endpoint": {"type": "boolean"},
                "edges": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "source": {"type": "string"},
                            "target": {"type": "string"},
                            "explanation": {"type": "string"}
                        },
                        "required": ["source", "target", "explanation"]
                    }
                }
            },
            "required": ["endpoint", "edges"],
            "additionalProperties": False
        }
    }
}

# Schema 4: EXTRACT_INFO - returns {endpoint, exist_nodes[], new_nodes[]}
EXTRACT_INFO_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "extract_info",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "endpoint": {"type": "boolean"},
                "exist_nodes": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "value": {"type": "string"},
                            "confidential_level": {"type": "number"}
                        },
                        "required": ["id", "value", "confidential_level"]
                    }
                },
                "new_nodes": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "description": {"type": "string"},
                            "weight": {"type": "number"},
                            "uncertainty": {"type": "number"},
                            "confidential_level": {"type": "number"},
                            "relevance": {"type": "number"},
                            "value": {"type": "string"}
                        },
                        "required": ["name", "description", "weight", "uncertainty", "confidential_level", "relevance", "value"]
                    }
                }
            },
            "required": ["endpoint", "exist_nodes", "new_nodes"],
            "additionalProperties": False
        }
    }
}

# Schema 5: UPDATE_GRAPH - returns {updates: [{id, name, weight, uncertainty, update_reason}]}
UPDATE_GRAPH_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "update_graph",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "updates": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "name": {"type": "string"},
                            "weight": {"type": "number"},
                            "uncertainty": {"type": "number"},
                            "update_reason": {"type": "string"}
                        },
                        "required": ["id", "name", "weight", "uncertainty", "update_reason"]
                    }
                }
            },
            "required": ["updates"],
            "additionalProperties": False
        }
    }
}