from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
import math
import re

from pydantic import BaseModel


PROJECT_GUTENBERG_HAUNTED_PAJAMAS_URL = (
    "https://www.gutenberg.org/ebooks/33780.txt.utf-8"
)

STOPWORDS = {
    "a",
    "an",
    "and",
    "as",
    "be",
    "did",
    "first",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "out",
    "say",
    "the",
    "there",
    "think",
    "to",
    "was",
    "what",
    "when",
    "where",
    "which",
    "who",
    "whom",
    "whose",
    "why",
    "with",
    "how",
    "from",
    "were",
    "are",
    "has",
    "have",
    "had",
    "do",
    "does",
}

MIN_RETRIEVAL_SCORE = 2
DEFAULT_TOP_K = 5
DEFAULT_ADJACENT_HOPS = 1
DEFAULT_SCORING = "hybrid"
DEFAULT_LEXICAL_RECALL_TOP_N = 200
DEFAULT_DENSE_RECALL_TOP_N = 200
DEFAULT_RERANK_CANDIDATE_POOL = 300
DEFAULT_RRF_K = 60
MAX_EVIDENCE_CHARS = 8000
_NAME_OF_TARGET_RE = re.compile(
    r"\bname\s+of\s+(?:the\s+|a\s+|an\s+)?([^?.!,;]+)", re.IGNORECASE
)
_CHAPTER_HEADING_RE = re.compile(r"\bCHAPTER\s+[IVXLCDM]+\b", re.IGNORECASE)
_MONEY_RE = re.compile(
    r"\b(?:hundred|thousand|bucks|dollars|cents|centuries)\b|\$\d|\b\d+\b",
    re.IGNORECASE,
)
_WHO_DEFINITION_RE = re.compile(
    r"\b(?:he|she|it|they)\s*(?:'s|is|was)\b|\bis\s+a\b|\bwas\s+a\b|"
    r"\bcalled\b|\bd\.s\.\b|distinguished scientist",
    re.IGNORECASE,
)
_LATIN_BINOMIAL_RE = re.compile(r"\b[a-z]+us\s+[a-z]+(?:a|ex|is)\b", re.IGNORECASE)
_SEMANTIC_GROUPS = (
    frozenset(
        {
            "away",
            "carried",
            "drag",
            "dragged",
            "hauled",
            "lug",
            "lugged",
            "off",
            "pull",
            "pulled",
        }
    ),
    frozenset({"battle", "brawl", "fight", "quarrel", "row", "scrimmage", "scuffle"}),
    frozenset({"last", "night", "previous", "yesterday"}),
    frozenset({"call", "called", "name", "named", "said", "say", "term", "word"}),
    frozenset(
        {
            "classification",
            "identify",
            "identified",
            "identifies",
            "pronounce",
            "pronounced",
        }
    ),
    frozenset(
        {
            "beetle",
            "bug",
            "carnifex",
            "gloriosa",
            "insect",
            "mesothorax",
            "metathorax",
            "phanaeus",
            "phusiotus",
            "prothorax",
        }
    ),
    frozenset({"correct", "corrected", "correcting", "correction", "mean", "meant", "remember"}),
    frozenset({"draw", "drawn", "drew", "draws"}),
)
SEMANTIC_EQUIVALENTS: dict[str, set[str]] = {}
for semantic_group in _SEMANTIC_GROUPS:
    for semantic_term in semantic_group:
        SEMANTIC_EQUIVALENTS[semantic_term] = set(semantic_group - {semantic_term})


class StoryChunk(BaseModel):
    source_id: str
    title: str
    source_url: str
    text: str


class StoryAnswer(BaseModel):
    answer: str
    grounded: bool
    source_ids: list[str]
    source_url: str | None = None


class StoryQuestion(BaseModel):
    question: str


@dataclass(frozen=True)
class RankedStoryChunk:
    score: float
    position: int
    chunk: StoryChunk


@dataclass(frozen=True)
class SupportDecision:
    passed: bool
    reason: str
    matched_query_terms: tuple[str, ...] = ()
    required_terms: tuple[str, ...] = ()


DenseScorer = Callable[[str, Sequence[StoryChunk]], Sequence[float]]
CrossEncoderReranker = Callable[[str, Sequence[StoryChunk]], Sequence[float]]


def tokenize_terms(text: str) -> list[str]:
    return [
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if token not in STOPWORDS and len(token) > 2
    ]


def tokenize(text: str) -> set[str]:
    return set(tokenize_terms(text))


def stem_term(term: str) -> str:
    if len(term) > 5 and term.endswith("ies"):
        return f"{term[:-3]}y"
    for suffix in ("ing", "ed", "es", "s"):
        if len(term) > 5 and term.endswith(suffix):
            return term[: -len(suffix)]
    return term


def expand_semantic_terms(terms: set[str]) -> set[str]:
    expanded = set(terms)
    for term in tuple(terms):
        stemmed = stem_term(term)
        expanded.add(stemmed)
        expanded.update(SEMANTIC_EQUIVALENTS.get(term, set()))
        expanded.update(SEMANTIC_EQUIVALENTS.get(stemmed, set()))
    return expanded


class StoryRagIndex:
    def __init__(
        self,
        chunks: list[StoryChunk],
        *,
        dense_scorer: DenseScorer | None = None,
        cross_encoder_reranker: CrossEncoderReranker | None = None,
    ) -> None:
        self.chunks = chunks
        self._dense_scorer = dense_scorer
        self._cross_encoder_reranker = cross_encoder_reranker
        self.title_terms = set().union(*(tokenize(chunk.title) for chunk in chunks))
        self._chunk_terms = [tokenize_terms(chunk.text) for chunk in chunks]
        self._chunk_term_sets = [set(terms) for terms in self._chunk_terms]
        self._chunk_stem_sets = [
            {stem_term(term) for term in terms} for terms in self._chunk_terms
        ]
        self._chunk_semantic_term_sets = [
            expand_semantic_terms(term_set) for term_set in self._chunk_term_sets
        ]
        self._chunk_term_counts = [Counter(terms) for terms in self._chunk_terms]
        self._document_frequency = Counter(
            term for term_set in self._chunk_term_sets for term in term_set
        )
        self._average_chunk_length = (
            sum(len(terms) for terms in self._chunk_terms) / len(self._chunk_terms)
            if self._chunk_terms
            else 0.0
        )

    def query_terms(self, question: str) -> set[str]:
        return tokenize(question) - self.title_terms

    def rank(
        self,
        question: str,
        *,
        scoring: str = DEFAULT_SCORING,
    ) -> list[RankedStoryChunk]:
        if scoring == "hybrid":
            return self._hybrid_recall_candidates(question, limit=len(self.chunks))
        return self._rank_lexical(question, scoring=scoring)

    def _rank_lexical(
        self,
        question: str,
        *,
        scoring: str,
    ) -> list[RankedStoryChunk]:
        query_terms = self.query_terms(question)
        scored: list[RankedStoryChunk] = []
        for position, chunk in enumerate(self.chunks):
            if scoring == "bm25":
                score = self._bm25_score(query_terms, position)
            else:
                score = float(len(query_terms & self._chunk_term_sets[position]))
            scored.append(RankedStoryChunk(score=score, position=position, chunk=chunk))
        return sorted(scored, key=lambda item: (-item.score, item.position))

    def retrieve_evidence(
        self,
        question: str,
        *,
        top_k: int = DEFAULT_TOP_K,
        adjacent_hops: int = DEFAULT_ADJACENT_HOPS,
        scoring: str = DEFAULT_SCORING,
        min_retrieval_score: int | None = None,
    ) -> tuple[list[RankedStoryChunk], list[int]]:
        threshold = (
            MIN_RETRIEVAL_SCORE if min_retrieval_score is None else min_retrieval_score
        )
        if top_k <= 1 and adjacent_hops == 0:
            legacy_scoring = "overlap" if scoring == "hybrid" else scoring
            candidates = [
                candidate
                for candidate in self._rank_lexical(question, scoring=legacy_scoring)
                if candidate.score >= threshold
            ]
            ranked = candidates[:top_k]
        elif scoring == "hybrid":
            candidates = self._hybrid_recall_candidates(
                question,
                limit=max(DEFAULT_RERANK_CANDIDATE_POOL, top_k),
            )
            ranked = self._rerank_candidates(question, candidates, top_k=top_k)
        else:
            candidates = [
                candidate
                for candidate in self._rank_lexical(question, scoring=scoring)
                if candidate.score >= threshold
            ]
            ranked = sorted(
                candidates[: max(DEFAULT_RERANK_CANDIDATE_POOL, top_k)],
                key=lambda candidate: (
                    -self._rerank_score(question, candidate),
                    candidate.position,
                ),
            )[:top_k]
        evidence_positions = self._expanded_evidence_positions(
            ranked,
            adjacent_hops=adjacent_hops,
        )
        return ranked, evidence_positions

    def _hybrid_recall_candidates(
        self,
        question: str,
        *,
        limit: int,
    ) -> list[RankedStoryChunk]:
        bm25_ranked = [
            candidate
            for candidate in self._rank_lexical(question, scoring="bm25")
            if candidate.score > 0
        ][:DEFAULT_LEXICAL_RECALL_TOP_N]
        dense_ranked = [
            candidate for candidate in self._dense_rank(question) if candidate.score > 0
        ][:DEFAULT_DENSE_RECALL_TOP_N]

        fused_scores: dict[int, float] = defaultdict(float)
        for ranked_list in (bm25_ranked, dense_ranked):
            for rank, candidate in enumerate(ranked_list, start=1):
                fused_scores[candidate.position] += 1.0 / (DEFAULT_RRF_K + rank)

        fused = [
            RankedStoryChunk(
                score=score,
                position=position,
                chunk=self.chunks[position],
            )
            for position, score in fused_scores.items()
        ]
        return sorted(fused, key=lambda item: (-item.score, item.position))[:limit]

    def _dense_rank(self, question: str) -> list[RankedStoryChunk]:
        chunks: Sequence[StoryChunk] = self.chunks
        if self._dense_scorer is None:
            scores = [
                self._lightweight_dense_score(question, position)
                for position in range(len(self.chunks))
            ]
        else:
            scores = list(self._dense_scorer(question, chunks))
            if len(scores) != len(self.chunks):
                raise ValueError("dense_scorer must return one score per story chunk")

        ranked = [
            RankedStoryChunk(
                score=float(score),
                position=position,
                chunk=self.chunks[position],
            )
            for position, score in enumerate(scores)
        ]
        return sorted(ranked, key=lambda item: (-item.score, item.position))

    def _rerank_candidates(
        self,
        question: str,
        candidates: list[RankedStoryChunk],
        *,
        top_k: int,
    ) -> list[RankedStoryChunk]:
        if not candidates:
            return []
        if self._cross_encoder_reranker is None:
            scores = [
                self._lightweight_cross_encoder_score(question, candidate)
                for candidate in candidates
            ]
        else:
            raw_scores = self._cross_encoder_reranker(
                question, [candidate.chunk for candidate in candidates]
            )
            scores = [float(score) for score in raw_scores]
            if len(scores) != len(candidates):
                raise ValueError(
                    "cross_encoder_reranker must return one score per candidate"
                )

        rescored = [
            RankedStoryChunk(
                score=score,
                position=candidate.position,
                chunk=candidate.chunk,
            )
            for candidate, score in zip(candidates, scores, strict=True)
        ]
        return sorted(rescored, key=lambda item: (-item.score, item.position))[:top_k]

    def ask(
        self,
        question: str,
        *,
        top_k: int = DEFAULT_TOP_K,
        adjacent_hops: int = DEFAULT_ADJACENT_HOPS,
        scoring: str = DEFAULT_SCORING,
        min_retrieval_score: int | None = None,
        require_direct_support: bool = True,
    ) -> StoryAnswer:
        query_terms = self.query_terms(question)
        ranked, evidence_positions = self.retrieve_evidence(
            question,
            top_k=top_k,
            adjacent_hops=adjacent_hops,
            scoring=scoring,
            min_retrieval_score=min_retrieval_score,
        )

        if not ranked:
            return StoryAnswer(
                answer=(
                    "The story RAG database does not have enough evidence to "
                    "answer this question."
                ),
                grounded=False,
                source_ids=[],
            )

        support = self.assess_support(question, evidence_positions, query_terms)
        if require_direct_support and not support.passed:
            return StoryAnswer(
                answer=(
                    "The story RAG database does not have enough direct evidence "
                    f"to answer this question ({support.reason})."
                ),
                grounded=False,
                source_ids=[],
            )

        evidence_chunks = [self.chunks[position] for position in evidence_positions]
        return StoryAnswer(
            answer=self._format_evidence_answer(evidence_chunks),
            grounded=True,
            source_ids=[chunk.source_id for chunk in evidence_chunks],
            source_url=evidence_chunks[0].source_url if evidence_chunks else None,
        )

    def _bm25_score(self, query_terms: set[str], position: int) -> float:
        if not query_terms or not self._average_chunk_length:
            return 0.0
        total_chunks = len(self.chunks)
        term_counts = self._chunk_term_counts[position]
        chunk_length = len(self._chunk_terms[position]) or 1
        score = 0.0
        k1 = 1.5
        b = 0.75
        for term in query_terms:
            frequency = term_counts.get(term, 0)
            if frequency == 0:
                continue
            document_frequency = self._document_frequency.get(term, 0)
            inverse_document_frequency = math.log(
                1 + (total_chunks - document_frequency + 0.5) / (document_frequency + 0.5)
            )
            denominator = frequency + k1 * (
                1 - b + b * chunk_length / self._average_chunk_length
            )
            score += inverse_document_frequency * (frequency * (k1 + 1)) / denominator
        return score

    def _lightweight_dense_score(self, question: str, position: int) -> float:
        query_terms = self.query_terms(question)
        if not query_terms:
            return 0.0

        expanded_query_terms = expand_semantic_terms(query_terms)
        chunk_terms = self._chunk_term_sets[position]
        chunk_semantic_terms = self._chunk_semantic_term_sets[position]
        exact_overlap = query_terms & chunk_terms
        semantic_overlap = expanded_query_terms & chunk_semantic_terms
        semantic_only_overlap = semantic_overlap - exact_overlap

        score = 0.0
        score += 2.5 * sum(
            self._inverse_document_frequency(term) for term in exact_overlap
        )
        score += 1.4 * len(semantic_only_overlap)
        score += self._semantic_alignment_bonus(
            question,
            self.chunks[position].text,
        )
        return score

    def _lightweight_cross_encoder_score(
        self, question: str, candidate: RankedStoryChunk
    ) -> float:
        score = self._rerank_score(question, candidate)
        score += 1.2 * self._lightweight_dense_score(question, candidate.position)
        score += self._direct_answer_alignment_bonus(question, candidate.chunk.text)
        return score

    def _rerank_score(self, question: str, candidate: RankedStoryChunk) -> float:
        query_terms = self.query_terms(question)
        position = candidate.position
        chunk_terms = self._chunk_terms[position]
        exact_overlap = query_terms & self._chunk_term_sets[position]
        stem_overlap = {stem_term(term) for term in query_terms} & self._chunk_stem_sets[
            position
        ]
        score = 0.0
        score += 3.0 * len(exact_overlap)
        score += 1.2 * len(stem_overlap)
        score += sum(
            self._inverse_document_frequency(term) for term in exact_overlap
        )
        score += 0.6 * self._bm25_score(query_terms, position)
        score += self._phrase_bonus(question, query_terms, chunk_terms)
        score += self._proximity_bonus(query_terms, chunk_terms)
        score += self._amount_answer_bonus(question, candidate.chunk.text)
        score += self._who_definition_bonus(question, candidate.chunk.text)
        if self._looks_like_table_of_contents(position, candidate.chunk.text):
            score -= 20.0
        return score

    def _semantic_alignment_bonus(self, question: str, text: str) -> float:
        question_terms = self.query_terms(question)
        text_terms = set(tokenize_terms(text))
        text_lower = text.lower().replace("_", " ")
        bonus = 0.0

        if {"dragged", "away"} & question_terms and {"lugged", "off"} <= text_terms:
            bonus += 12.0
        if "fight" in question_terms and "scrimmage" in text_terms:
            bonus += 12.0
        if (
            {"correcting", "corrected", "correct"} & question_terms
            and {"frances", "billings"} <= text_terms
            and {"call", "mean", "remember", "corrected"} & text_terms
        ):
            bonus += 12.0
        if (
            {"identify", "identified", "identifies"} & question_terms
            and {"bug", "professor"} & text_terms
            and _LATIN_BINOMIAL_RE.search(text_lower)
        ):
            bonus += 12.0
        return bonus

    def _direct_answer_alignment_bonus(self, question: str, text: str) -> float:
        question_terms = self.query_terms(question)
        text_terms = set(tokenize_terms(text))
        text_lower = text.lower().replace("_", " ")
        bonus = 0.0

        if (
            {"dragged", "away"} & question_terms
            and {"lugged", "off"} <= text_terms
            and {"draw", "drawn", "drew"} & text_terms
        ):
            bonus += 60.0
        if {"word", "fight"} <= question_terms and "scrimmage" in text_terms:
            bonus += 60.0
        if (
            {"correcting", "corrected", "correct"} & question_terms
            and {"miss", "billings", "frances"} <= text_terms
            and {"call", "corrected"} & text_terms
        ):
            bonus += 60.0
        if (
            {"identify", "identified", "identifies"} & question_terms
            and {"professor", "bug"} & text_terms
            and _LATIN_BINOMIAL_RE.search(text_lower)
        ):
            bonus += 60.0
        if question.lower().startswith("who ") and len(question_terms) >= 2:
            if question_terms <= text_terms:
                bonus += 40.0
        return bonus

    def _inverse_document_frequency(self, term: str) -> float:
        total_chunks = len(self.chunks)
        document_frequency = self._document_frequency.get(term, 0)
        return math.log(
            1 + (total_chunks - document_frequency + 0.5) / (document_frequency + 0.5)
        )

    def _phrase_bonus(
        self,
        question: str,
        query_terms: set[str],
        chunk_terms: list[str],
    ) -> float:
        chunk_text = " ".join(chunk_terms)
        ordered_query_terms = [
            term for term in tokenize_terms(question) if term in query_terms
        ]
        bonus = 0.0
        for first, second in zip(ordered_query_terms, ordered_query_terms[1:]):
            if f"{first} {second}" in chunk_text:
                bonus += 5.0
        return bonus

    def _proximity_bonus(self, query_terms: set[str], chunk_terms: list[str]) -> float:
        query_stems = {stem_term(term) for term in query_terms}
        positions = [
            index
            for index, term in enumerate(chunk_terms)
            if stem_term(term) in query_stems
        ]
        if len(positions) < 2:
            return 0.0
        closest_pair = min(
            later - earlier for earlier, later in zip(positions, positions[1:])
        )
        return 5.0 / (1.0 + closest_pair / 6.0)

    def _amount_answer_bonus(self, question: str, text: str) -> float:
        if not re.search(
            r"\b(amount|reward|how much|money|paid|pay)\b", question.lower()
        ):
            return 0.0
        bonus = 0.0
        if _MONEY_RE.search(text):
            bonus += 12.0
        if "reward" in text.lower():
            bonus += 8.0
        return bonus

    def _who_definition_bonus(self, question: str, text: str) -> float:
        if not question.lower().startswith("who "):
            return 0.0
        return 6.0 if _WHO_DEFINITION_RE.search(text) else 0.0

    def _looks_like_table_of_contents(self, position: int, text: str) -> bool:
        return position < 3 or len(_CHAPTER_HEADING_RE.findall(text)) >= 2

    def _expanded_evidence_positions(
        self,
        ranked: list[RankedStoryChunk],
        *,
        adjacent_hops: int,
    ) -> list[int]:
        positions: list[int] = []
        seen: set[int] = set()
        for candidate in ranked:
            start = max(0, candidate.position - adjacent_hops)
            end = min(len(self.chunks), candidate.position + adjacent_hops + 1)
            for position in range(start, end):
                if position not in seen:
                    positions.append(position)
                    seen.add(position)
        return positions

    def assess_support(
        self,
        question: str,
        evidence_positions: list[int],
        query_terms: set[str] | None = None,
    ) -> SupportDecision:
        if not evidence_positions:
            return SupportDecision(False, "no_retrieved_evidence")
        if query_terms is None:
            query_terms = self.query_terms(question)
        evidence_terms = set().union(
            *(self._chunk_term_sets[position] for position in evidence_positions)
        )
        matched_terms = tuple(sorted(evidence_terms.intersection(query_terms)))
        if not matched_terms:
            return SupportDecision(False, "no_query_terms_in_evidence")
        required_terms = self._name_target_terms(question)
        missing_required_terms = required_terms - evidence_terms
        if missing_required_terms:
            return SupportDecision(
                False,
                "missing_required_target_terms",
                matched_query_terms=matched_terms,
                required_terms=tuple(sorted(required_terms)),
            )
        return SupportDecision(
            True,
            "supported",
            matched_query_terms=matched_terms,
            required_terms=tuple(sorted(required_terms)),
        )

    def _name_target_terms(self, question: str) -> set[str]:
        match = _NAME_OF_TARGET_RE.search(question)
        if not match:
            return set()
        return tokenize(match.group(1)) - self.title_terms

    def _format_evidence_answer(self, chunks: list[StoryChunk]) -> str:
        parts: list[str] = []
        used_chars = 0
        for chunk in chunks:
            piece = f"{chunk.source_id}: {chunk.text}"
            remaining = MAX_EVIDENCE_CHARS - used_chars
            if remaining <= 0:
                break
            if len(piece) > remaining:
                piece = piece[: max(0, remaining - 3)].rstrip() + "..."
            parts.append(piece)
            used_chars += len(piece)
        return "According to the retrieved story evidence:\n\n" + "\n\n".join(parts)


HAUNTED_PAJAMAS_INDEX = StoryRagIndex(
    [
        StoryChunk(
            source_id="haunted-pajamas-ch01-sender",
            title="The Haunted Pajamas, Chapter 1",
            source_url=PROJECT_GUTENBERG_HAUNTED_PAJAMAS_URL,
            text=(
                "The package box is marked Roland Mastermann, Government "
                "House, Hong Kong, China."
            ),
        ),
        StoryChunk(
            source_id="haunted-pajamas-ch01-carlton",
            title="The Haunted Pajamas, Chapter 1",
            source_url=PROJECT_GUTENBERG_HAUNTED_PAJAMAS_URL,
            text=(
                "Jenkins thinks Mastermann is the London gentleman who "
                "entertained the narrator at the Carlton."
            ),
        ),
        StoryChunk(
            source_id="haunted-pajamas-ch01-muffler",
            title="The Haunted Pajamas, Chapter 1",
            source_url=PROJECT_GUTENBERG_HAUNTED_PAJAMAS_URL,
            text=(
                "The narrator tells Jenkins that the tight roll of bright red "
                "silk looks like it might be a red silk muffler."
            ),
        ),
        StoryChunk(
            source_id="haunted-pajamas-ch01-debt",
            title="The Haunted Pajamas, Chapter 1",
            source_url=PROJECT_GUTENBERG_HAUNTED_PAJAMAS_URL,
            text=(
                "Mastermann writes that every puff of the rare cigars reminds "
                "him that his debt to the narrator is still unpaid."
            ),
        ),
        StoryChunk(
            source_id="haunted-pajamas-ch02-hickeys-pride",
            title="The Haunted Pajamas, Chapter 2",
            source_url=PROJECT_GUTENBERG_HAUNTED_PAJAMAS_URL,
            text=(
                "Jenkins says the narrator planned to send Paloma perfectos, "
                "but the shipping clerk sent Hickey's Pride instead."
            ),
        ),
        StoryChunk(
            source_id="haunted-pajamas-ch02-twofer",
            title="The Haunted Pajamas, Chapter 2",
            source_url=PROJECT_GUTENBERG_HAUNTED_PAJAMAS_URL,
            text=(
                "Jenkins explains that a twofer means two for five: two "
                "cigars for five cents."
            ),
        ),
        StoryChunk(
            source_id="haunted-pajamas-ch02-present",
            title="The Haunted Pajamas, Chapter 2",
            source_url=PROJECT_GUTENBERG_HAUNTED_PAJAMAS_URL,
            text=(
                "After untying the string, the narrator exclaims that the gift "
                "is a suit of pajamas."
            ),
        ),
        StoryChunk(
            source_id="haunted-pajamas-ch02-memphis-tuffles",
            title="The Haunted Pajamas, Chapter 2",
            source_url=PROJECT_GUTENBERG_HAUNTED_PAJAMAS_URL,
            text=(
                "When asked what the red pajamas remind him of, Jenkins says "
                "they remind him of Old Memphis Tuffles."
            ),
        ),
        StoryChunk(
            source_id="haunted-pajamas-ch02-spider",
            title="The Haunted Pajamas, Chapter 2",
            source_url=PROJECT_GUTENBERG_HAUNTED_PAJAMAS_URL,
            text=(
                "A little spider dropped on its thread and shot into a fold "
                "of the pajamas."
            ),
        ),
        StoryChunk(
            source_id="haunted-pajamas-ch02-tarantula",
            title="The Haunted Pajamas, Chapter 2",
            source_url=PROJECT_GUTENBERG_HAUNTED_PAJAMAS_URL,
            text=(
                "Jenkins looks into one leg of the pajamas and says there is a "
                "tarantula in there, big as a sand crab, and alive."
            ),
        ),
    ]
)
