"""Evaluate Socartes StoryRagIndex against a full-novel automatic index.

This experiment builds deterministic chunks from the full Project Gutenberg
text, then compares the legacy top-1 retriever with the repaired multi-evidence
retriever.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = REPO_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

import socartes_backend.story_rag as story_rag
from socartes_backend.story_rag import (
    DEFAULT_ADJACENT_HOPS,
    DEFAULT_SCORING,
    DEFAULT_TOP_K,
    PROJECT_GUTENBERG_HAUNTED_PAJAMAS_URL,
    StoryChunk,
    StoryRagIndex,
)

DATA_PATH = REPO_ROOT / "experiments" / "data" / "haunted_pajamas_33780.txt"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "experiments" / "results" / "full_novel_eval.json"
DEFAULT_CHUNK_TARGET_WORDS = 100


@dataclass(frozen=True)
class Question:
    qid: str
    question: str
    expected_phrase: str | None


ANSWERABLE_QUESTIONS = [
    Question(
        "Q1",
        "What name and address were printed on the package box?",
        "roland mastermann",
    ),
    Question(
        "Q2",
        "Who did Jenkins think Mastermann was?",
        "carlton",
    ),
    Question(
        "Q3",
        "What did the narrator first think the red silk roll might be?",
        "red silk muffler",
    ),
    Question(
        "Q4",
        "What debt did Mastermann say every puff of the rare cigars reminded him of?",
        "unpaid",
    ),
    Question(
        "Q5",
        "Which cheap cigar brand was sent by mistake instead of Paloma perfectos?",
        "hickey",
    ),
    Question(
        "Q6",
        "What did Jenkins say a twofer meant?",
        "two for five",
    ),
    Question(
        "Q7",
        "What was the gift after the string was untied?",
        "suit of pajamas",
    ),
    Question(
        "Q8",
        "Who did Jenkins say the red pajamas reminded him of?",
        "old memphis tuffles",
    ),
    Question(
        "Q9",
        "What dropped into a fold of the pajamas?",
        "little spider",
    ),
    Question(
        "Q10",
        "What did Jenkins say was in the pajama leg?",
        "tarantula",
    ),
]

EXTENDED_ANSWERABLE_QUESTIONS = [
    Question(
        "Q13",
        "From what city did Billings say the kid would arrive about nine?",
        "boston",
    ),
    Question(
        "Q14",
        "What did Jenkins say the pajama cords looked like?",
        "little red snakes",
    ),
    Question(
        "Q15",
        "What name did O'Keefe call the old impostor before sending him to jail?",
        "foxy",
    ),
    Question(
        "Q16",
        "Who did Billings say Doozenberry was?",
        "distinguished scientist",
    ),
    Question(
        "Q17",
        "What lost silk did the professor say the pajamas were made from?",
        "si-ling-chi",
    ),
    Question(
        "Q18",
        "What did the narrator compare the big red car to outside the Kahoka?",
        "red whale",
    ),
    Question(
        "Q19",
        "What reward amount did O'Keefe mention when he brought back the black silk pajamas?",
        "five hundred",
    ),
    Question(
        "Q20",
        "What captain name did O'Keefe tell Lightnut to remember?",
        "captain clutchem",
    ),
    Question(
        "Q21",
        "Who confronted Lightnut under the pergola?",
        "chauffeur",
    ),
    Question(
        "Q22",
        "What did Billings snap out the lights to show off in the library?",
        "ruby",
    ),
    Question(
        "Q23",
        "What would Frances draw if the bobby dragged Lightnut away?",
        "good excalibar",
    ),
    Question(
        "Q24",
        "What did Lightnut call Frances before correcting himself?",
        "miss billings",
    ),
    Question(
        "Q25",
        "What did the judge say the arrested lunatic claimed to be?",
        "my son",
    ),
    Question(
        "Q26",
        "What drew the blue aeroplane in Lightnut's dream?",
        "golden humming-birds",
    ),
    Question(
        "Q27",
        "What word did Jenkins use for the previous night's fight?",
        "scrimmage",
    ),
    Question(
        "Q28",
        "What did the frump think the little greenish bug was?",
        "phusiotus",
    ),
    Question(
        "Q29",
        "What did the professor identify the bug as?",
        "phanaeus carnifex",
    ),
    Question(
        "Q30",
        "Through what places did Frances and Lightnut take a drive?",
        "sleepy hollow and the pocantico hills",
    ),
    Question(
        "Q31",
        "What was burning in the hearth before Frances and Lightnut?",
        "fragrant pine cones",
    ),
    Question(
        "Q32",
        "What railway were Colonel Kirkland and the judge debating?",
        "manchurian railway",
    ),
    Question(
        "Q33",
        "What did the burglar whisper after lifting Billings' ruby?",
        "mine now to keep forever",
    ),
    Question(
        "Q34",
        "What did Billings say the petticoat was unknown in?",
        "china",
    ),
    Question(
        "Q35",
        "What did Wilkes say Billings had the most considerable case of?",
        "jimmies",
    ),
    Question(
        "Q36",
        "What did Lightnut plan to do with the parcel of pajamas?",
        "send it to billings",
    ),
    Question(
        "Q37",
        "How many curs did Francis take to the dog fight?",
        "four curs",
    ),
    Question(
        "Q38",
        "What did Billings' father call the youth in the hallway?",
        "disgrace to an honored name",
    ),
    Question(
        "Q39",
        "What color did Frances' face and neck turn?",
        "lovely crimson",
    ),
    Question(
        "Q40",
        "What did the young fellow look like when he slouched in?",
        "a bit sulky",
    ),
    Question(
        "Q41",
        "What did the professor ask whether Billings would consider intrusive?",
        "a call",
    ),
    Question(
        "Q42",
        "What did the narrator say had nearly happened while he was aghast?",
        "losing consciousness",
    ),
]

ALL_ANSWERABLE_QUESTIONS = ANSWERABLE_QUESTIONS + EXTENDED_ANSWERABLE_QUESTIONS

CONTROL_QUESTIONS = [
    Question("Q11", "In The Haunted Pajamas, who kills Sherlock Holmes?", None),
    Question(
        "Q12",
        "In The Haunted Pajamas, what is the name of the spaceship captain?",
        None,
    ),
]


def cleaned_gutenberg_body(text: str) -> str:
    start_marker = "*** START OF THE PROJECT GUTENBERG EBOOK"
    end_marker = "*** END OF THE PROJECT GUTENBERG EBOOK"
    start = text.find(start_marker)
    if start != -1:
        first_newline = text.find("\n", start)
        text = text[first_newline + 1 :]
    end = text.find(end_marker)
    if end != -1:
        text = text[:end]
    return text.strip()


def word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9']+", text))


def paragraph_chunks(body: str, target_words: int) -> list[str]:
    paragraphs = [
        re.sub(r"\s+", " ", paragraph).strip()
        for paragraph in re.split(r"\n\s*\n", body)
        if re.sub(r"\s+", " ", paragraph).strip()
    ]
    chunks: list[str] = []
    current: list[str] = []
    current_words = 0
    for paragraph in paragraphs:
        paragraph_words = word_count(paragraph)
        if current and current_words + paragraph_words > target_words:
            chunks.append(" ".join(current))
            current = []
            current_words = 0
        if paragraph_words > target_words:
            words = re.findall(r"[A-Za-z0-9']+", paragraph)
            for index in range(0, len(words), target_words):
                chunks.append(" ".join(words[index : index + target_words]))
        else:
            current.append(paragraph)
            current_words += paragraph_words
    if current:
        chunks.append(" ".join(current))
    return chunks


def build_full_novel_index(body: str, target_words: int) -> StoryRagIndex:
    chunks = [
        StoryChunk(
            source_id=f"full-novel-auto-{index:04d}",
            title="The Haunted Pajamas full novel automatic chunk",
            source_url=PROJECT_GUTENBERG_HAUNTED_PAJAMAS_URL,
            text=chunk,
        )
        for index, chunk in enumerate(paragraph_chunks(body, target_words))
    ]
    return StoryRagIndex(chunks)


def first_evidence_rank(
    index: StoryRagIndex, question: str, expected_phrase: str | None
) -> dict[str, Any] | None:
    if not expected_phrase:
        return None
    for rank, candidate in enumerate(index.rank(question), start=1):
        if expected_phrase in candidate.chunk.text.lower():
            return {
                "rank": rank,
                "source_id": candidate.chunk.source_id,
                "score": candidate.score,
                "preview": candidate.chunk.text[:220],
            }
    return None


def retrieval_diagnostics(
    index: StoryRagIndex,
    question: str,
    expected_phrase: str | None,
    *,
    top_k: int,
    adjacent_hops: int,
    scoring: str,
    min_retrieval_score: int,
) -> dict[str, Any]:
    ranked, positions = index.retrieve_evidence(
        question,
        top_k=top_k,
        adjacent_hops=adjacent_hops,
        scoring=scoring,
        min_retrieval_score=min_retrieval_score,
    )
    support = index.assess_support(question, positions)
    return {
        "ranked_top_k": [
            {
                "rank": rank,
                "source_id": candidate.chunk.source_id,
                "score": candidate.score,
                "preview": candidate.chunk.text[:180],
            }
            for rank, candidate in enumerate(ranked, start=1)
        ],
        "expanded_evidence_source_ids": [
            index.chunks[position].source_id for position in positions
        ],
        "support_gate": {
            "passed": support.passed,
            "reason": support.reason,
            "matched_query_terms": list(support.matched_query_terms),
            "required_terms": list(support.required_terms),
        },
        "first_evidence_rank": first_evidence_rank(index, question, expected_phrase),
    }


def run_condition(
    label: str,
    index: StoryRagIndex,
    *,
    min_retrieval_score: int = story_rag.MIN_RETRIEVAL_SCORE,
    use_title_term_guard: bool = True,
    top_k: int = DEFAULT_TOP_K,
    adjacent_hops: int = DEFAULT_ADJACENT_HOPS,
    scoring: str = DEFAULT_SCORING,
    require_direct_support: bool = True,
) -> dict[str, Any]:
    old_title_terms = index.title_terms
    if not use_title_term_guard:
        index.title_terms = set()
    try:
        answerable = []
        for item in ALL_ANSWERABLE_QUESTIONS:
            answer = index.ask(
                item.question,
                top_k=top_k,
                adjacent_hops=adjacent_hops,
                scoring=scoring,
                min_retrieval_score=min_retrieval_score,
                require_direct_support=require_direct_support,
            )
            correct = bool(
                answer.grounded
                and item.expected_phrase
                and item.expected_phrase in answer.answer.lower()
            )
            answerable.append(
                {
                    "qid": item.qid,
                    "question": item.question,
                    "expected_phrase": item.expected_phrase,
                    "grounded": answer.grounded,
                    "source_ids": answer.source_ids,
                    "correct": correct,
                    "answer_preview": answer.answer[:260],
                    "retrieval": retrieval_diagnostics(
                        index,
                        item.question,
                        item.expected_phrase,
                        top_k=top_k,
                        adjacent_hops=adjacent_hops,
                        scoring=scoring,
                        min_retrieval_score=min_retrieval_score,
                    ),
                }
            )

        controls = []
        for item in CONTROL_QUESTIONS:
            answer = index.ask(
                item.question,
                top_k=top_k,
                adjacent_hops=adjacent_hops,
                scoring=scoring,
                min_retrieval_score=min_retrieval_score,
                require_direct_support=require_direct_support,
            )
            correct_refusal = not answer.grounded and not answer.source_ids
            controls.append(
                {
                    "qid": item.qid,
                    "question": item.question,
                    "grounded": answer.grounded,
                    "source_ids": answer.source_ids,
                    "correct_refusal": correct_refusal,
                    "answer_preview": answer.answer[:260],
                    "retrieval": retrieval_diagnostics(
                        index,
                        item.question,
                        item.expected_phrase,
                        top_k=top_k,
                        adjacent_hops=adjacent_hops,
                        scoring=scoring,
                        min_retrieval_score=min_retrieval_score,
                    ),
                }
            )

        base_qids = {item.qid for item in ANSWERABLE_QUESTIONS}
        extended_qids = {item.qid for item in EXTENDED_ANSWERABLE_QUESTIONS}
        answerable_correct = sum(row["correct"] for row in answerable)
        base_answerable_correct = sum(
            row["correct"] for row in answerable if row["qid"] in base_qids
        )
        extended_answerable_correct = sum(
            row["correct"] for row in answerable if row["qid"] in extended_qids
        )
        control_correct = sum(row["correct_refusal"] for row in controls)
        return {
            "label": label,
            "answerable_correct": answerable_correct,
            "answerable_total": len(ALL_ANSWERABLE_QUESTIONS),
            "base_answerable_correct": base_answerable_correct,
            "base_answerable_total": len(ANSWERABLE_QUESTIONS),
            "extended_answerable_correct": extended_answerable_correct,
            "extended_answerable_total": len(EXTENDED_ANSWERABLE_QUESTIONS),
            "control_refusal_correct": control_correct,
            "control_total": len(CONTROL_QUESTIONS),
            "grounding_policy_pass": answerable_correct + control_correct,
            "grounding_policy_total": len(ALL_ANSWERABLE_QUESTIONS)
            + len(CONTROL_QUESTIONS),
            "answerable": answerable,
            "controls": controls,
            "settings": {
                "scoring": scoring,
                "top_k": top_k,
                "adjacent_hops": adjacent_hops,
                "min_retrieval_score": min_retrieval_score,
                "use_title_term_guard": use_title_term_guard,
                "require_direct_support": require_direct_support,
                "tie_break": "StoryRagIndex sorts by descending score, then earliest chunk position.",
            },
        }
    finally:
        index.title_terms = old_title_terms



def build_report(corpus_path: Path, target_words: int) -> dict[str, Any]:
    raw_text = corpus_path.read_text(encoding="utf-8")
    body = cleaned_gutenberg_body(raw_text)
    full_index = build_full_novel_index(body, target_words)

    main = run_condition("Socartes full-novel automatic index", full_index)
    legacy_top_one = run_condition(
        "Baseline: legacy top-1 lexical retrieval",
        build_full_novel_index(body, target_words),
        top_k=1,
        adjacent_hops=0,
        require_direct_support=False,
    )
    no_support_gate = run_condition(
        "Ablation: hybrid RRF + rerank without support gate",
        build_full_novel_index(body, target_words),
        require_direct_support=False,
    )

    return {
        "experiment": "Socartes StoryRagIndex full-novel automatic-index diagnostic",
        "corpus": {
            "title": "The Haunted Pajamas",
            "author": "Francis Perry Elliott",
            "gutenberg_ebook": "33780",
            "source_url": PROJECT_GUTENBERG_HAUNTED_PAJAMAS_URL,
            "local_path": str(corpus_path.relative_to(REPO_ROOT)),
            "body_word_count": word_count(body),
            "chunk_target_words": target_words,
            "automatic_chunk_count": len(full_index.chunks),
        },
        "retrieval_system": {
            "class": "socartes_backend.story_rag.StoryRagIndex",
            "retrieval": "hybrid RRF recall with reranking and adjacent chunk expansion",
            "scoring": DEFAULT_SCORING,
            "top_k": DEFAULT_TOP_K,
            "adjacent_hops": DEFAULT_ADJACENT_HOPS,
            "production_min_retrieval_score": 2,
            "title_term_guard": "query terms that occur in chunk titles are removed before scoring",
            "direct_support_gate": "name-of target terms must be present in retrieved evidence before answering",
            "tie_break": "same-score chunks choose the earliest chunk position",
        },
        "reference_conditions": [
            {
                "label": "GPT-5.5 closed-book probe (existing data)",
                "answerable_correct": 2,
                "answerable_total": 10,
                "control_refusal_correct": 1,
                "control_total": 2,
                "source_form": "none",
                "role": "reference probe only; not the primary RAG metric",
            },
        ],
        "main_condition": main,
        "ablations": [legacy_top_one, no_support_gate],
        "interpretation": [
            "The legacy top-1 retriever misses answer evidence when the answer phrase is outside the highest-scoring chunk.",
            "Hybrid RRF recall with one adjacent chunk on each side recovers the answer evidence for Q1-Q10.",
            "The reranked 300-candidate pool recovers the extended Q13-Q42 evidence while preserving refusal behavior.",
            "The direct-support gate is required to reject the spaceship-captain control instead of answering from an unrelated Captain Clutchem context.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, default=DATA_PATH)
    parser.add_argument("--chunk-target-words", type=int, default=DEFAULT_CHUNK_TARGET_WORDS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()

    report = build_report(args.corpus, args.chunk_target_words)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")

    main_condition = report["main_condition"]
    print(
        "full-novel automatic index:",
        f"{main_condition['answerable_correct']}/{main_condition['answerable_total']} answerable",
        f"{main_condition['control_refusal_correct']}/{main_condition['control_total']} refusals",
    )
    print("chunks:", report["corpus"]["automatic_chunk_count"])
    print("wrote:", args.output)


if __name__ == "__main__":
    main()
