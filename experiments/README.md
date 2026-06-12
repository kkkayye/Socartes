# Socartes RAG full-novel diagnostic

This experiment evaluates `StoryRagIndex` against a full-novel automatic index and compares the legacy top-1 retriever with the repaired multi-evidence retriever.

## What changed

- Main experiment: build a deterministic automatic index from the full Project Gutenberg text of *The Haunted Pajamas*.
- Questions: evaluate 42 diagnostic questions: Q1-Q10 original answerable questions, Q13-Q42 extension answerable questions, and Q11-Q12 unsupported controls.
- Retriever: hybrid RRF recall with BM25 plus real vector recall, reranking, `top_k=5`, `adjacent_hops=1`, title-term guard, `MIN_RETRIEVAL_SCORE=2`, and a direct-support refusal gate.
- Embedding layer: `StoryVectorScorer` uses `qwen3-embedding-0.6b` through an OpenAI-compatible `/embeddings` endpoint when an embedding API key is available. It can fall back to a cached local `all-MiniLM-L6-v2` ONNX model, then to the deterministic hashed baseline for dependency-free reproduction.
- Vector store: `sqlite-vec` is used when the optional `vector` extra is installed; otherwise the same vectors are searched with deterministic in-memory cosine search.

## Reproduce

```bash
python experiments/full_novel_eval.py
```

The script reads `experiments/data/haunted_pajamas_33780.txt` and writes:

```bash
experiments/results/full_novel_eval.json
```

Default chunking uses paragraph accumulation with `--chunk-target-words 100`. With the committed Gutenberg text this produces 1047 automatic chunks.

## Current result

| Condition | Original Q1-Q10 | New Q13-Q42 | Overall answerable | Correct refusal Q11-Q12 | Source form |
| --- | ---: | ---: | ---: | ---: | --- |
| GPT-5.5 closed-book probe | 2/10 | not tested | 2/10 | 1/2 | none |
| Socartes full-novel automatic index | 10/10 | 30/30 | 40/40 | 2/2 | chunk ID |

## Ablation result

| Configuration | Original Q1-Q10 | New Q13-Q42 | Overall answerable | Correct refusal Q11-Q12 |
| --- | ---: | ---: | ---: | ---: |
| Baseline: legacy top-1 lexical retrieval | 7/10 | 15/30 | 22/40 | 1/2 |
| Hybrid RRF + rerank without support gate | 10/10 | 30/30 | 40/40 | 1/2 |
| Full setting: hybrid RRF + rerank plus support gate | 10/10 | 30/30 | 40/40 | 2/2 |

## Interpretation


The legacy top-1 baseline misses Q2, Q5, and Q10 in the original set and reaches only 22/40 across all answerable questions because many answer passages are lower-ranked, adjacent, or semantically related rather than exact lexical matches. Hybrid RRF recall combines BM25 and vector candidates before reranking, recovering the full Q1-Q10 and Q13-Q42 answerable set. The support gate then fixes Q12 by refusing to answer the spaceship-captain question from the unrelated Captain Clutchem police context.
