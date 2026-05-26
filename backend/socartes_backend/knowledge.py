from .models import RetrievalChunk, ToolResult


KNOWLEDGE_BASE = [
    RetrievalChunk(
        source_id="rag-index-18",
        title="Learning systems index",
        content=(
            "RAG systems ground generated answers in retrieved external "
            "references so learners can inspect the source of a claim."
        ),
        confidence="medium",
    ),
    RetrievalChunk(
        source_id="workflow-note-01",
        title="Agent workflow brief",
        content=(
            "Multi-agent learning systems separate planning, execution, and "
            "critique to make task ownership and revision steps visible."
        ),
        confidence="high",
    ),
    RetrievalChunk(
        source_id="mcp-tool-07",
        title="Tool adapter note",
        content=(
            "MCP-style adapters expose external APIs, databases, and file "
            "systems through controlled contracts that can be audited."
        ),
        confidence="high",
    ),
]


def default_tool_results(goal: str) -> list[ToolResult]:
    return [
        ToolResult(
            adapter="external_api",
            action="fetch_domain_state",
            output=f"Attached live-domain context for goal: {goal}",
        ),
        ToolResult(
            adapter="knowledge_database",
            action="query_indexed_notes",
            output="Returned ranked notes for RAG, MCP tool use, and agent review.",
        ),
        ToolResult(
            adapter="filesystem",
            action="read_learner_artifacts",
            output="Loaded scoped study artifacts for learner-specific context.",
        ),
    ]
