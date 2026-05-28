from fastapi.testclient import TestClient

from socartes_backend.app import app


def test_health_endpoint_reports_backend_identity():
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "socartes-backend",
        "version": "0.1.0",
    }


def test_learn_endpoint_returns_traceable_agent_output():
    client = TestClient(app)

    response = client.post(
        "/api/v1/learn",
        json={
            "goal": "Explain how planner, executor, and critic agents work together.",
            "learner_context": "Prefer concise explanations.",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["plan"]["agent"] == "planner"
    assert payload["review"]["status"] == "approved"
    assert payload["final_answer"]
    assert payload["retrieved_context"]
    assert payload["tool_results"]
    assert payload["reflection_events"]


def test_agents_endpoint_documents_each_backend_worker():
    client = TestClient(app)

    response = client.get("/api/v1/agents")

    assert response.status_code == 200
    payload = response.json()
    assert "planner" in payload["agents"]
    assert "executor" in payload["agents"]
    assert "critic" in payload["agents"]


def test_story_rag_endpoint_returns_grounded_source_id():
    client = TestClient(app)

    response = client.post(
        "/api/v1/story-rag/ask",
        json={"question": "What did Jenkins say was in the pajama leg?"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["grounded"] is True
    assert payload["source_ids"] == ["haunted-pajamas-ch01-tarantula"]
    assert "tarantula" in payload["answer"].lower()
