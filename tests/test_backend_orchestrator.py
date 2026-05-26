from socartes_backend.agents import SocartesOrchestrator


def test_orchestrator_runs_full_agentic_learning_cycle():
    orchestrator = SocartesOrchestrator()

    trace = orchestrator.run(
        goal="Compare RAG agents with MCP tool-using agents for a research workflow.",
        learner_context="The learner wants a concise, citation-backed explanation.",
    )

    assert trace.plan.agent == "planner"
    assert [task.owner for task in trace.plan.tasks] == [
        "planner",
        "retriever",
        "executor",
        "critic",
    ]
    assert trace.retrieved_context
    assert {chunk.source_id for chunk in trace.retrieved_context} >= {
        "rag-index-18",
        "workflow-note-01",
    }
    assert {result.adapter for result in trace.tool_results} == {
        "external_api",
        "knowledge_database",
        "filesystem",
    }
    assert trace.draft.citations
    assert trace.review.agent == "critic"
    assert trace.review.status == "approved"
    assert trace.reflection_events
    assert trace.reflection_events[-1].event_type == "planner_update"
    assert "RAG" in trace.final_answer
    assert "MCP" in trace.final_answer


def test_agent_catalog_exposes_role_boundaries():
    orchestrator = SocartesOrchestrator()

    catalog = orchestrator.agent_catalog()

    assert set(catalog) >= {"planner", "executor", "critic", "retriever", "tool_adapter"}
    assert catalog["planner"]["responsibility"].startswith("Convert learner goals")
    assert "acceptance criteria" in catalog["critic"]["checks"]
