"""cls_backend — the retrieval, caching, and carrier sub-package.

This package contains everything below the service layer (``cls_service.py``).
None of these modules import Streamlit, Chainlit, or FastAPI — they are pure
business logic and are independently testable.

Module map
----------
pipeline.py     Core RAG+CAG engine: embedding, vector search, lexical retrieval,
                CAG cache look-up, extractive answer assembly.  The public contract
                is ``instant_answer(query, collection, cache, encoder, ...)``.

dllm.py         Carrier cleanup helpers: sparse activation gate, constrained
                correction prompts, number-grounding checks.  Only activates when
                the extracted text contains obvious PDF/OCR artefacts.

cag_cache.py    Semantic CAG layer: a second ChromaDB collection keyed by
                question embeddings.  Near-identical repeat queries skip the
                Evidence Store lookup and return cached evidence directly.

readers.py      Document loaders for PDF (PyMuPDF), DOCX, TXT, CSV, and JSON.
                Each loader returns ``list[tuple[page_number, text]]`` so the
                rest of the pipeline is document-type-agnostic.

spectrum.py     Presentation-only: classifies queries into four semantic
                categories (Contacts / Procedure / Specs / General) and
                returns the colour, glyph, and CSS for the answer card.
                Never touches retrieval or embeddings.

query_repair.py Pre-retrieval query normaliser: strips conversational scaffolding
                and filler words before the text hits the embedding model.
"""
