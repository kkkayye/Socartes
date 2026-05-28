from fastapi import FastAPI

from .agents import SocartesOrchestrator
from .models import StudyRequest, StudyTrace
from .story_rag import HAUNTED_PAJAMAS_INDEX, StoryAnswer, StoryQuestion

VERSION = "0.1.0"

app = FastAPI(
    title="Socartes Backend",
    version=VERSION,
    description=(
        "Standalone backend prototype for Socartes multi-agent planning, RAG, "
        "MCP-style tool use, and reflection."
    ),
)
orchestrator = SocartesOrchestrator()


@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "service": "socartes-backend",
        "version": VERSION,
    }


@app.get("/api/v1/agents")
def agents() -> dict[str, dict[str, dict[str, str]]]:
    return {"agents": orchestrator.agent_catalog()}


@app.post("/api/v1/learn", response_model=StudyTrace)
def learn(request: StudyRequest) -> StudyTrace:
    return orchestrator.run(
        goal=request.goal,
        learner_context=request.learner_context,
    )


@app.post("/api/v1/story-rag/ask", response_model=StoryAnswer)
def ask_story_rag(request: StoryQuestion) -> StoryAnswer:
    return HAUNTED_PAJAMAS_INDEX.ask(request.question)
