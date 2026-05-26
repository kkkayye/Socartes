from typing import Literal

from pydantic import BaseModel, Field


AgentOwner = Literal["planner", "retriever", "executor", "critic"]
ToolAdapter = Literal["external_api", "knowledge_database", "filesystem"]
Confidence = Literal["low", "medium", "high"]
ReviewStatus = Literal["approved", "revision_required"]


class StudyRequest(BaseModel):
    goal: str = Field(..., min_length=3)
    learner_context: str = ""


class PlanTask(BaseModel):
    id: str
    owner: AgentOwner
    objective: str
    evidence_required: list[str] = Field(default_factory=list)
    acceptance_criteria: list[str] = Field(default_factory=list)


class AgentPlan(BaseModel):
    agent: Literal["planner"] = "planner"
    summary: str
    tasks: list[PlanTask]


class RetrievalChunk(BaseModel):
    source_id: str
    title: str
    content: str
    confidence: Confidence


class ToolResult(BaseModel):
    adapter: ToolAdapter
    action: str
    output: str
    safe: bool = True


class DraftAnswer(BaseModel):
    agent: Literal["executor"] = "executor"
    content: str
    citations: list[str]
    tool_results_used: list[str]
    open_gaps: list[str] = Field(default_factory=list)


class CriticIssue(BaseModel):
    type: str
    claim: str
    instruction: str


class CriticReview(BaseModel):
    agent: Literal["critic"] = "critic"
    status: ReviewStatus
    checks: list[str]
    issues: list[CriticIssue] = Field(default_factory=list)
    approved: bool


class ReflectionEvent(BaseModel):
    event_type: str
    agent: str
    message: str


class StudyTrace(BaseModel):
    goal: str
    learner_context: str
    plan: AgentPlan
    retrieved_context: list[RetrievalChunk]
    tool_results: list[ToolResult]
    draft: DraftAnswer
    review: CriticReview
    reflection_events: list[ReflectionEvent]
    final_answer: str
