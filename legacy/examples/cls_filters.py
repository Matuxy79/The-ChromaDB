PRISM_FILTERS = {
    "purple": {"field": "colour_code", "operator": "equals", "value": "purple"},
    "green":  {"field": "colour_code", "operator": "equals", "value": "green"},
    "blue":   {"field": "colour_code", "operator": "equals", "value": "blue"},
    "orange": {"field": "colour_code", "operator": "equals", "value": "orange"},
    "yellow": {"field": "colour_code", "operator": "equals", "value": "yellow"},
}

def prism_to_metadata_filter(lane: str | None) -> dict | None:
    """None means no filter (unfiltered debug mode). Re-derive every query."""
    return PRISM_FILTERS.get(lane) if lane else None
