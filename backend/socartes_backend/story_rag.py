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
    "with",
}


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


def tokenize(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if token not in STOPWORDS and len(token) > 2
    }


class StoryRagIndex:
    def __init__(self, chunks: list[StoryChunk]) -> None:
        self.chunks = chunks

    def ask(self, question: str) -> StoryAnswer:
        query_terms = tokenize(question)
        scored_chunks = [
            (len(query_terms & tokenize(chunk.text + " " + chunk.title)), chunk)
            for chunk in self.chunks
        ]
        score, chunk = max(scored_chunks, key=lambda item: item[0], default=(0, None))

        if chunk is None or score == 0:
            return StoryAnswer(
                answer=(
                    "The story RAG database does not have enough evidence to "
                    "answer this question."
                ),
                grounded=False,
                source_ids=[],
            )

        return StoryAnswer(
            answer=f"According to {chunk.source_id}: {chunk.text}",
            grounded=True,
            source_ids=[chunk.source_id],
            source_url=chunk.source_url,
        )


HAUNTED_PAJAMAS_INDEX = StoryRagIndex(
    [
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
            source_id="haunted-pajamas-ch01-present",
            title="The Haunted Pajamas, Chapter 1",
            source_url=PROJECT_GUTENBERG_HAUNTED_PAJAMAS_URL,
            text=(
                "After untying the string, the narrator exclaims that the gift "
                "is a suit of pajamas."
            ),
        ),
        StoryChunk(
            source_id="haunted-pajamas-ch01-tarantula",
            title="The Haunted Pajamas, Chapter 1",
            source_url=PROJECT_GUTENBERG_HAUNTED_PAJAMAS_URL,
            text=(
                "Jenkins looks into one leg of the pajamas and says there is a "
                "tarantula in there, big as a sand crab, and alive."
            ),
        ),
    ]
)
