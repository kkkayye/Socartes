# Source-Grounded RAG vs Direct GPT-5.5 on Obscure Fiction QA

## Abstract

This report compares two ways of answering obscure plot questions: a direct, closed-book query to `gpt-5.5` and the Socartes Story RAG endpoint backed by a local evidence index. The test uses Francis Perry Elliott's public-domain novel *The Haunted Pajamas* from Project Gutenberg. The evaluation was expanded from 4 probes to 12 probes: 10 answerable plot-detail questions and 2 no-evidence control questions. Direct `gpt-5.5` answered 3 of 12 correctly, while Socartes RAG answered all 10 source-supported questions with citations and refused both unsupported controls.

![Comparison chart](gpt55-vs-rag-haunted-pajamas.svg)

## Research Question

Can a small source-grounded RAG system outperform a direct model query when the task depends on obscure fiction details that are unlikely to be reliably memorized by the model?

## Recommended Sample Size

For this diagnostic, 12 questions is a better minimum than 10. Ten answerable questions are enough to cover several detail types, while two control questions test refusal behavior. This keeps the report readable while reducing the chance that one lucky answer dominates the result.

## Corpus

The corpus is Project Gutenberg EBook #33780, *The Haunted Pajamas* by Francis Perry Elliott:

- Book page: <https://www.gutenberg.org/ebooks/33780>
- Plain text source: <https://www.gutenberg.org/ebooks/33780.txt.utf-8>

The Socartes RAG index uses local chunks from Chapters 1 and 2:

| Source ID | Evidence target |
| --- | --- |
| `haunted-pajamas-ch01-sender` | The package is marked Roland Mastermann, Government House, Hong Kong, China. |
| `haunted-pajamas-ch01-carlton` | Jenkins identifies Mastermann as the London gentleman from the Carlton. |
| `haunted-pajamas-ch01-muffler` | The narrator first thinks the red silk roll may be a red silk muffler. |
| `haunted-pajamas-ch01-debt` | Mastermann says every puff of the cigars reminds him of an unpaid debt. |
| `haunted-pajamas-ch02-hickeys-pride` | Hickey's Pride was sent instead of Paloma perfectos. |
| `haunted-pajamas-ch02-twofer` | Jenkins explains that a twofer means two for five. |
| `haunted-pajamas-ch02-present` | After the string is untied, the gift is revealed as a suit of pajamas. |
| `haunted-pajamas-ch02-memphis-tuffles` | Jenkins says the red pajamas remind him of Old Memphis Tuffles. |
| `haunted-pajamas-ch02-spider` | A little spider drops into a fold of the pajamas. |
| `haunted-pajamas-ch02-tarantula` | Jenkins says there is a tarantula in the pajama leg. |

## Method

Both systems were asked the same 12 questions.

Direct GPT-5.5 condition:

- Model: `gpt-5.5`
- Prompt rule: answer from model knowledge only, without browsing or database access.
- Output format: JSON answers with a low/medium/high confidence field.

Socartes RAG condition:

- Endpoint behavior: retrieve a matching source chunk, return a cited answer, or refuse if the database has no evidence.
- Retrieval guard: title terms such as "Haunted" and "Pajamas" are removed from query scoring, and a chunk must match at least two body evidence terms. This prevents a question like "In The Haunted Pajamas, who kills Sherlock Holmes?" from matching a chunk only because the title appears in the question.

## Questions and Ground Truth

| ID | Question | Ground truth |
| --- | --- | --- |
| Q1 | What name and address were printed on the package box? | Roland Mastermann, Government House, Hong Kong, China. |
| Q2 | Who did Jenkins think Mastermann was? | The London gentleman who entertained the narrator at the Carlton. |
| Q3 | What did the narrator first think the red silk roll might be? | A red silk muffler. |
| Q4 | What debt did every puff of the rare cigars remind Mastermann of? | His debt to the narrator was still unpaid. |
| Q5 | Which cheap cigar brand was sent by mistake instead of Paloma perfectos? | Hickey's Pride. |
| Q6 | What did Jenkins say a twofer meant? | Two for five, or two cigars for five cents. |
| Q7 | What was the gift after the string was untied? | A suit or pair of pajamas. |
| Q8 | Who did Jenkins say the red pajamas reminded him of? | Old Memphis Tuffles. |
| Q9 | What dropped into a fold of the pajamas? | A little spider. |
| Q10 | What did Jenkins say was in the pajama leg? | A tarantula, big as a sand crab, and alive. |
| Q11 | Who kills Sherlock Holmes? | No supporting evidence in this story RAG database; correct behavior is refusal. |
| Q12 | What is the name of the spaceship captain? | No supporting evidence in this story RAG database; correct behavior is refusal. |

## Results

| ID | Direct GPT-5.5 answer | Direct score | Socartes RAG behavior | RAG score |
| --- | --- | --- | --- | --- |
| Q1 | Uncertain; did not know the exact name/address. | Incorrect/non-answer | Cites `haunted-pajamas-ch01-sender`. | Correct, grounded |
| Q2 | Uncertain; did not know who Jenkins thought Mastermann was. | Incorrect/non-answer | Cites `haunted-pajamas-ch01-carlton`. | Correct, grounded |
| Q3 | Uncertain; guessed a red silk garment/fabric. | Incorrect | Cites `haunted-pajamas-ch01-muffler`. | Correct, grounded |
| Q4 | Uncertain; did not know the specific debt. | Incorrect/non-answer | Cites `haunted-pajamas-ch01-debt`. | Correct, grounded |
| Q5 | Uncertain; did not know the cheap cigar brand. | Incorrect/non-answer | Cites `haunted-pajamas-ch02-hickeys-pride`. | Correct, grounded |
| Q6 | Two cigars for five cents. | Correct | Cites `haunted-pajamas-ch02-twofer`. | Correct, grounded |
| Q7 | A pair of red silk pajamas. | Correct | Cites `haunted-pajamas-ch02-present`. | Correct, grounded |
| Q8 | The devil. | Incorrect | Cites `haunted-pajamas-ch02-memphis-tuffles`. | Correct, grounded |
| Q9 | Uncertain; did not know exactly what dropped into the fold. | Incorrect/non-answer | Cites `haunted-pajamas-ch02-spider`. | Correct, grounded |
| Q10 | Uncertain; did not know what Jenkins said was in the leg. | Incorrect/non-answer | Cites `haunted-pajamas-ch02-tarantula`. | Correct, grounded |
| Q11 | Uncertain between Professor Moriarty and Arthur Conan Doyle. | Incorrect/control failure | Refuses because no source evidence exists. | Correct refusal |
| Q12 | There is no spaceship captain in the novel. | Correct but uncited | Refuses because no source evidence exists. | Correct refusal |

## Metrics

| Metric | Direct GPT-5.5 | Socartes RAG |
| --- | ---: | ---: |
| Answer accuracy | 25% | 100% |
| Source-grounding policy compliance | 0% | 100% |
| Answer failure rate | 75% | 0% |

Metric definitions:

- Answer accuracy counts exact answers on answerable questions and correct no-evidence handling on controls. Q12 was counted as correct for direct `gpt-5.5`, but it was still uncited.
- Source-grounding policy compliance means every response is either supported by a source ID or explicitly refused because no source evidence exists.
- Answer failure means an incorrect answer, a non-answer for an answerable question, or an unsupported external answer on a control question.

## Interpretation

The direct model query was conservative on several obscure details and often admitted uncertainty. That is better than fabricating every answer, but it still failed the learner's task because most answerable plot questions remained unanswered. It also guessed incorrectly on Old Memphis Tuffles and produced external Sherlock Holmes candidates for a control question that should be rejected under a source-grounded policy.

Socartes RAG behaved like a source-bound answerer. It returned exact local source IDs for answerable questions and refused both unsupported controls. The result demonstrates the practical value of RAG for learning workflows where the important requirement is not general fluency, but verifiable alignment with a supplied corpus.

## Limitations

This is a diagnostic evaluation, not a broad benchmark. It uses one public-domain novel, 12 hand-written questions, and a deterministic RAG index. Direct model outputs may vary across runs. The result is still useful for Socartes because it isolates the core product claim: when a learner asks about a specific corpus, the system should prefer database-grounded evidence over model memory.

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
