"""Ontology tagging. MeSH heading extraction (`mesh.py`) is done -- captured directly from
Europe PMC metadata at ingest time (see ingest/bulk_match.py), not inferred. EDAM + coarse
domain-science mapping remain a later, optional enhancement layered on top of the MeSH headings
already present -- see ROADMAP.md Phase 2 for the planned interface
(`edam_mapper.map_to_edam`, `domain_mapper.map_to_domain`) and acceptance criteria."""
