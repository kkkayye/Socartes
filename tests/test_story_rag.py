import sys
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("SOCARTES_EMBEDDING_PROVIDER", "hash")

from socartes_backend.story_rag import (
    HAUNTED_PAJAMAS_INDEX,
    OpenAIEmbeddingModel,
    PROJECT_GUTENBERG_HAUNTED_PAJAMAS_URL,
    StoryChunk,
    StoryRagIndex,
)
from experiments.full_novel_eval import (
    ANSWERABLE_QUESTIONS,
    CONTROL_QUESTIONS,
    DATA_PATH,
    EXTENDED_ANSWERABLE_QUESTIONS,
    build_full_novel_index,
    cleaned_gutenberg_body,
)


def haunted_pajamas_test_index() -> StoryRagIndex:
    return StoryRagIndex(
        [
            StoryChunk(
                source_id="haunted-pajamas-ch01-muffler",
                title="The Haunted Pajamas, Chapter 1",
                source_url=PROJECT_GUTENBERG_HAUNTED_PAJAMAS_URL,
                text=(
                    "The narrator tells Jenkins that the tight roll of bright "
                    "red silk looks like it might be a red silk muffler."
                ),
            ),
            StoryChunk(
                source_id="haunted-pajamas-ch01-present",
                title="The Haunted Pajamas, Chapter 1",
                source_url=PROJECT_GUTENBERG_HAUNTED_PAJAMAS_URL,
                text=(
                    "After untying the string, the narrator exclaims that the "
                    "gift is a suit of pajamas."
                ),
            ),
            StoryChunk(
                source_id="haunted-pajamas-ch01-tarantula",
                title="The Haunted Pajamas, Chapter 1",
                source_url=PROJECT_GUTENBERG_HAUNTED_PAJAMAS_URL,
                text=(
                    "Jenkins looks into one leg of the pajamas and says there "
                    "is a tarantula in there, big as a sand crab, and alive."
                ),
            ),
        ]
    )


def test_story_rag_uses_vector_backend_for_dense_recall():
    def embedding_model(texts):
        vectors = []
        for text in texts:
            if "Mastermann lugged me off" in text or "Who dragged" in text:
                vectors.append([1.0, 0.0])
            else:
                vectors.append([0.0, 1.0])
        return vectors

    index = StoryRagIndex(
        [
            StoryChunk(
                source_id="lexical-decoy",
                title="Test",
                source_url="memory://test",
                text="The narrator was dragged away by a rumor about the hallway.",
            ),
            StoryChunk(
                source_id="semantic-answer",
                title="Test",
                source_url="memory://test",
                text="Mastermann lugged me off before I could speak.",
            ),
        ],
        vector_backend="memory",
        embedding_model=embedding_model,
    )

    lexical_top = index.rank(
        "Who dragged the narrator away?",
        scoring="bm25",
    )[0]
    dense_top = index._dense_rank("Who dragged the narrator away?")[0]

    assert index.vector_backend_name == "memory"
    assert lexical_top.chunk.source_id == "lexical-decoy"
    assert dense_top.chunk.source_id == "semantic-answer"


def test_openai_embedding_model_uses_text_embedding_3_large(monkeypatch):
    captured_request = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self):
            return (
                b'{"data":[{"index":0,"embedding":[3.0,4.0]},'
                b'{"index":1,"embedding":[0.0,5.0]}]}'
            )

    def fake_urlopen(req, timeout):
        captured_request["url"] = req.full_url
        captured_request["body"] = req.data.decode("utf-8")
        captured_request["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr("socartes_backend.story_rag.request.urlopen", fake_urlopen)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    model = OpenAIEmbeddingModel(base_url="https://embeddings.example.test/v1")
    vectors = model(["first", "second"])

    assert model.model_name == "text-embedding-3-large"
    assert captured_request["url"] == "https://embeddings.example.test/v1/embeddings"
    assert '"model": "text-embedding-3-large"' in captured_request["body"]
    assert captured_request["timeout"] == 60.0
    assert vectors == [[0.6, 0.8], [0.0, 1.0]]


def test_story_rag_answers_obscure_plot_questions_from_database_chunks():
    index = haunted_pajamas_test_index()

    muffler_answer = index.ask(
        "What did the narrator first think the red silk roll might be?"
    )
    assert muffler_answer.grounded is True
    assert "haunted-pajamas-ch01-muffler" in muffler_answer.source_ids
    assert "red silk muffler" in muffler_answer.answer.lower()

    present_answer = index.ask("What was the gift after the string was untied?")
    assert present_answer.grounded is True
    assert "haunted-pajamas-ch01-present" in present_answer.source_ids
    assert "pajamas" in present_answer.answer.lower()

    tarantula_answer = index.ask("What did Jenkins say was in the pajama leg?")
    assert tarantula_answer.grounded is True
    assert "haunted-pajamas-ch01-tarantula" in tarantula_answer.source_ids
    assert "tarantula" in tarantula_answer.answer.lower()
    assert "sand crab" in tarantula_answer.answer.lower()


def test_story_rag_refuses_when_database_has_no_plot_evidence():
    index = haunted_pajamas_test_index()

    answer = index.ask("Who kills Sherlock Holmes in this story?")

    assert answer.grounded is False
    assert answer.source_ids == []
    assert "not have enough" in answer.answer


def test_story_rag_refuses_unrelated_question_even_when_title_is_named():
    index = haunted_pajamas_test_index()

    answer = index.ask("In The Haunted Pajamas, who kills Sherlock Holmes?")

    assert answer.grounded is False
    assert answer.source_ids == []
    assert "not have enough" in answer.answer


def test_story_rag_covers_expanded_twelve_question_evaluation_set():
    answerable_questions = [
        (
            "What name and address were printed on the package box?",
            "haunted-pajamas-ch01-sender",
            "roland mastermann",
        ),
        (
            "Who did Jenkins think Mastermann was?",
            "haunted-pajamas-ch01-carlton",
            "carlton",
        ),
        (
            "What did the narrator first think the red silk roll might be?",
            "haunted-pajamas-ch01-muffler",
            "red silk muffler",
        ),
        (
            "What debt did Mastermann say every puff of the rare cigars reminded him of?",
            "haunted-pajamas-ch01-debt",
            "still unpaid",
        ),
        (
            "Which cheap cigar brand was sent by mistake instead of Paloma perfectos?",
            "haunted-pajamas-ch02-hickeys-pride",
            "hickey's pride",
        ),
        (
            "What did Jenkins say a twofer meant?",
            "haunted-pajamas-ch02-twofer",
            "two for five",
        ),
        (
            "What was the gift after the string was untied?",
            "haunted-pajamas-ch02-present",
            "suit of pajamas",
        ),
        (
            "Who did Jenkins say the red pajamas reminded him of?",
            "haunted-pajamas-ch02-memphis-tuffles",
            "old memphis tuffles",
        ),
        (
            "What dropped into a fold of the pajamas?",
            "haunted-pajamas-ch02-spider",
            "little spider",
        ),
        (
            "What did Jenkins say was in the pajama leg?",
            "haunted-pajamas-ch02-tarantula",
            "tarantula",
        ),
    ]

    for question, source_id, expected_text in answerable_questions:
        answer = HAUNTED_PAJAMAS_INDEX.ask(question)
        assert answer.grounded is True, question
        assert source_id in answer.source_ids, question
        assert expected_text in answer.answer.lower(), question

    control_questions = [
        "In The Haunted Pajamas, who kills Sherlock Holmes?",
        "In The Haunted Pajamas, what is the name of the spaceship captain?",
    ]

    for question in control_questions:
        answer = HAUNTED_PAJAMAS_INDEX.ask(question)
        assert answer.grounded is False, question
        assert answer.source_ids == [], question


def test_story_rag_full_novel_index_uses_top_k_neighbors_and_refuses_controls():
    body = cleaned_gutenberg_body(DATA_PATH.read_text(encoding="utf-8"))
    index = build_full_novel_index(body, target_words=100)

    for item in ANSWERABLE_QUESTIONS:
        answer = index.ask(item.question)
        assert answer.grounded is True, item.question
        assert item.expected_phrase in answer.answer.lower(), item.question
        assert answer.source_ids, item.question

    for item in CONTROL_QUESTIONS:
        answer = index.ask(item.question)
        assert answer.grounded is False, item.question
        assert answer.source_ids == [], item.question


def test_story_rag_full_novel_index_recovers_extended_failure_questions():
    body = cleaned_gutenberg_body(DATA_PATH.read_text(encoding="utf-8"))
    index = build_full_novel_index(body, target_words=100)
    formerly_failed_qids = {"Q16", "Q17", "Q19", "Q21", "Q22"}

    for item in EXTENDED_ANSWERABLE_QUESTIONS:
        if item.qid not in formerly_failed_qids:
            continue

        answer = index.ask(item.question)

        assert answer.grounded is True, item.question
        assert item.expected_phrase in answer.answer.lower(), item.question
        assert answer.source_ids, item.question


def test_story_rag_hybrid_pipeline_recovers_second_extension_failure_questions():
    body = cleaned_gutenberg_body(DATA_PATH.read_text(encoding="utf-8"))
    index = build_full_novel_index(body, target_words=100)
    formerly_failed_qids = {"Q23", "Q24", "Q27", "Q29"}

    for item in EXTENDED_ANSWERABLE_QUESTIONS:
        if item.qid not in formerly_failed_qids:
            continue

        answer = index.ask(item.question)

        assert answer.grounded is True, item.question
        assert item.expected_phrase in answer.answer.lower(), item.question
        assert answer.source_ids, item.question


def test_story_rag_hybrid_pipeline_recovers_third_extension_questions():
    body = cleaned_gutenberg_body(DATA_PATH.read_text(encoding="utf-8"))
    index = build_full_novel_index(body, target_words=100)
    newest_qids = {f"Q{number}" for number in range(33, 43)}

    for item in EXTENDED_ANSWERABLE_QUESTIONS:
        if item.qid not in newest_qids:
            continue

        answer = index.ask(item.question)

        assert answer.grounded is True, item.question
        assert item.expected_phrase in answer.answer.lower(), item.question
        assert answer.source_ids, item.question


def test_story_rag_support_gate_blocks_unrelated_captain_name_false_hit():
    body = cleaned_gutenberg_body(DATA_PATH.read_text(encoding="utf-8"))
    index = build_full_novel_index(body, target_words=100)
    question = "In The Haunted Pajamas, what is the name of the spaceship captain?"

    unguarded_answer = index.ask(question, require_direct_support=False)
    assert unguarded_answer.grounded is True
    assert unguarded_answer.source_ids

    guarded_answer = index.ask(question)
    assert guarded_answer.grounded is False
    assert guarded_answer.source_ids == []
    assert "missing_required_target_terms" in guarded_answer.answer
