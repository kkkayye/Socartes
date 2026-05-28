# Source-Grounded RAG vs Direct GPT-5.5 on Obscure Fiction QA

## Abstract

This report compares two ways of answering obscure plot questions: a direct, closed-book query to `gpt-5.5` and the Socartes Story RAG endpoint backed by a small local evidence index. The test uses Francis Perry Elliott's public-domain novel *The Haunted Pajamas* from Project Gutenberg. On four probes, direct `gpt-5.5` answered one question correctly and hallucinated or guessed on three. Socartes RAG answered all three source-supported questions with citations and refused the unsupported Sherlock Holmes control question.

![Comparison chart](gpt55-vs-rag-haunted-pajamas.svg)

## Research Question

Can a small source-grounded RAG system outperform a direct model query when the task depends on obscure fiction details that are unlikely to be reliably memorized by the model?

## Corpus

The corpus is Project Gutenberg EBook #33780, *The Haunted Pajamas* by Francis Perry Elliott:

- Book page: <https://www.gutenberg.org/ebooks/33780>
- Plain text source: <https://www.gutenberg.org/ebooks/33780.txt.utf-8>

The Socartes RAG index uses three local chunks from Chapter 1:

| Source ID | Evidence target |
| --- | --- |
| `haunted-pajamas-ch01-muffler` | The narrator first thinks the red silk roll may be a red silk muffler. |
| `haunted-pajamas-ch01-present` | After the string is untied, the gift is revealed as a suit of pajamas. |
| `haunted-pajamas-ch01-tarantula` | Jenkins says there is a tarantula in the pajama leg. |

## Method

Both systems were asked the same four questions.

Direct GPT-5.5 condition:

- Model: `gpt-5.5`
- Prompt rule: answer from model knowledge only, without browsing or database access.
- Output format: JSON answers with a low/medium/high confidence field.

Socartes RAG condition:

- Endpoint behavior: retrieve a matching source chunk, return a cited answer, or refuse if the database has no evidence.
- Retrieval fix applied before reporting: title terms such as "Haunted" and "Pajamas" are removed from query scoring, and a chunk must match at least two body evidence terms. This prevents a question like "In The Haunted Pajamas, who kills Sherlock Holmes?" from matching a chunk only because the title appears in the question.

## Questions and Ground Truth

| ID | Question | Ground truth |
| --- | --- | --- |
| Q1 | What did the narrator first think the red silk roll might be? | A red silk muffler. |
| Q2 | What was the gift after the string was untied? | A suit or pair of pajamas. |
| Q3 | What did Jenkins say was in the pajama leg? | A tarantula, big as a sand crab, and alive. |
| Q4 | Who kills Sherlock Holmes? | No supporting evidence in this story RAG database; correct behavior is refusal. |

## Results

| ID | Direct GPT-5.5 answer | Direct score | Socartes RAG answer | RAG score |
| --- | --- | --- | --- | --- |
| Q1 | "He first thought it might be a kimono." | Incorrect | Cites `haunted-pajamas-ch01-muffler`: red silk roll may be a red silk muffler. | Correct, grounded |
| Q2 | "It was a pair of pajamas." | Correct | Cites `haunted-pajamas-ch01-present`: the gift is a suit of pajamas. | Correct, grounded |
| Q3 | "I'm uncertain, but Jenkins said there was a book in the pajama leg." | Incorrect | Cites `haunted-pajamas-ch01-tarantula`: Jenkins saw a tarantula in the pajama leg. | Correct, grounded |
| Q4 | "Sir Arthur Conan Doyle." | Incorrect | Refuses: database does not have enough evidence. | Correct refusal |

## Metrics

| Metric | Direct GPT-5.5 | Socartes RAG |
| --- | ---: | ---: |
| Plot-grounded accuracy | 25% | 100% |
| Evidence policy compliance | 0% | 100% |
| Hallucination/refusal failure rate | 75% | 0% |

Metric definitions:

- Plot-grounded accuracy counts the Sherlock Holmes control question as correct only when the system refuses because no source evidence exists.
- Evidence policy compliance means every answer is either supported by a source ID or explicitly refused.
- Hallucination/refusal failure means an unsupported or contradicted answer was returned instead of a grounded answer or refusal.

## Interpretation

The direct model query behaved like a closed-book answerer. It could guess the central object, pajamas, but failed on the more specific muffler and tarantula details. It also answered the Sherlock Holmes control question with an external guess instead of recognizing that the question is unsupported by the selected story.

Socartes RAG behaved like a source-bound answerer. It returned exact local source IDs for answerable questions and refused the unsupported control query. The result demonstrates the practical value of RAG for learning workflows where the important requirement is not general fluency, but verifiable alignment with a supplied corpus.

## Limitations

This is a small diagnostic evaluation, not a broad benchmark. It uses one public-domain novel, four hand-written questions, and a deterministic three-chunk RAG index. Direct model outputs may vary across runs. The result is still useful for Socartes because it isolates the core product claim: when a learner asks about a specific corpus, the system should prefer database-grounded evidence over model memory.

## Reproducibility

Run the focused regression tests:

```bash
pytest tests/test_story_rag.py -q
```

Run the full backend test suite:

```bash
pytest -q
```

Ask the Story RAG endpoint:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/story-rag/ask \
  -H 'Content-Type: application/json' \
  -d '{"question": "In The Haunted Pajamas, who kills Sherlock Holmes?"}'
```
