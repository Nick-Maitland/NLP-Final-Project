# Dense Validation Report

## Summary

- Status: `skipped`
- Generated at: `2026-04-20T15:42:55.080429+00:00`
- Model: `sentence-transformers/all-MiniLM-L6-v2`
- Collection: `ragfaq_dense_validation`
- Question: `What is self-attention?`
- Expected top-k: `3`
- Reason: sentence_transformers import unavailable: ImportError: tokenizers>=0.11.1,!=0.11.3,<0.14 is required for a normal functioning of this module, but found tokenizers==0.22.1. Try: pip install transformers -U or pip install -e '.[dev]' if you're working with git main
- Outcome: The dense path did not validate successfully in this environment.

## Check Results

| Step | Status | Detail |
| --- | --- | --- |
| Probe chromadb import | passed | chromadb imported successfully. |
| Probe sentence_transformers import | skipped | sentence_transformers import unavailable: ImportError: tokenizers>=0.11.1,!=0.11.3,<0.14 is required for a normal functioning of this module, but found tokenizers==0.22.1. Try: pip install transformers -U or pip install -e '.[dev]' if you're working with git main |
| Verify local MiniLM cache presence | not_run |  |
| Load MiniLM with local_files_only=True | not_run |  |
| Load repo documents and chunks | not_run |  |
| Prepare Chroma validation collection | not_run |  |
| Store chunks with collection.add(...) | not_run |  |
| Verify stored chunk count | not_run |  |
| Query Chroma with n_results=3 | not_run |  |
| Verify exactly top-3 chunks are returned | not_run |  |
| Verify source references are returned | not_run |  |
