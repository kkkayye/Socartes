from socartes_backend.story_rag import (
    HAUNTED_PAJAMAS_INDEX,
    PROJECT_GUTENBERG_HAUNTED_PAJAMAS_URL,
    StoryChunk,
    StoryRagIndex,
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


def test_story_rag_answers_obscure_plot_questions_from_database_chunks():
    index = haunted_pajamas_test_index()

    muffler_answer = index.ask(
        "What did the narrator first think the red silk roll might be?"
    )
    assert muffler_answer.grounded is True
    assert muffler_answer.source_ids == ["haunted-pajamas-ch01-muffler"]
    assert "red silk muffler" in muffler_answer.answer.lower()

    present_answer = index.ask("What was the gift after the string was untied?")
    assert present_answer.grounded is True
    assert present_answer.source_ids == ["haunted-pajamas-ch01-present"]
    assert "pajamas" in present_answer.answer.lower()

    tarantula_answer = index.ask("What did Jenkins say was in the pajama leg?")
    assert tarantula_answer.grounded is True
    assert tarantula_answer.source_ids == ["haunted-pajamas-ch01-tarantula"]
    assert "tarantula" in tarantula_answer.answer.lower()
    assert "sand crab" in tarantula_answer.answer.lower()


def test_story_rag_refuses_when_database_has_no_plot_evidence():
    index = haunted_pajamas_test_index()

    answer = index.ask("Who kills Sherlock Holmes in this story?")

    assert answer.grounded is False
    assert answer.source_ids == []
    assert "not have enough evidence" in answer.answer


def test_story_rag_refuses_unrelated_question_even_when_title_is_named():
    index = haunted_pajamas_test_index()

    answer = index.ask("In The Haunted Pajamas, who kills Sherlock Holmes?")

    assert answer.grounded is False
    assert answer.source_ids == []
    assert "not have enough evidence" in answer.answer


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
        assert answer.source_ids == [source_id], question
        assert expected_text in answer.answer.lower(), question

    control_questions = [
        "In The Haunted Pajamas, who kills Sherlock Holmes?",
        "In The Haunted Pajamas, what is the name of the spaceship captain?",
    ]

    for question in control_questions:
        answer = HAUNTED_PAJAMAS_INDEX.ask(question)
        assert answer.grounded is False, question
        assert answer.source_ids == [], question
