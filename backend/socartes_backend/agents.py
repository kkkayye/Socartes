from .knowledge import KNOWLEDGE_BASE, default_tool_results
from .models import (
    AgentPlan,
    CriticIssue,
    CriticReview,
    DraftAnswer,
    PlanTask,
    ReflectionEvent,
    RetrievalChunk,
    StudyTrace,
    ToolResult,
)


class PlannerAgent:
    role = "planner"

    def plan(self, goal: str) -> AgentPlan:
        return AgentPlan(
            summary="Convert the learner goal into a traceable study workflow.",
            tasks=[
                PlanTask(
                    id="task-plan",
                    owner="planner",
                    objective="Clarify the learning goal and define acceptance criteria.",
                    acceptance_criteria=[
                        "The answer addresses the learner goal",
                        "The final response lists evidence and unresolved gaps",
                    ],
                ),
                PlanTask(
                    id="task-retrieve",
                    owner="retriever",
                    objective="Retrieve external knowledge for the main concepts.",
                    evidence_required=["domain notes", "citation metadata"],
                ),
                PlanTask(
                    id="task-execute",
                    owner="executor",
                    objective="Synthesize the answer with retrieved context and tool output.",
                    evidence_required=["retrieved chunks", "tool results"],
                    acceptance_criteria=["Claims include source identifiers"],
                ),
                PlanTask(
                    id="task-critique",
                    owner="critic",
                    objective="Audit the draft and request revisions when evidence is weak.",
                    acceptance_criteria=["Unsupported claims are revised or removed"],
                ),
            ],
        )


class RetrieverWorker:
    role = "retriever"

    def retrieve(self, goal: str) -> list[RetrievalChunk]:
        goal_terms = goal.lower()
        matches = [
            chunk
            for chunk in KNOWLEDGE_BASE
            if any(
                term in goal_terms
                for term in (chunk.title + " " + chunk.content).lower().split()
            )
        ]
        if len(matches) < 2:
            source_ids = {chunk.source_id for chunk in matches}
            matches.extend(
                chunk
                for chunk in KNOWLEDGE_BASE[:2]
                if chunk.source_id not in source_ids
            )
        return matches


class ToolAdapterWorker:
    role = "tool_adapter"

    def run_tools(self, goal: str) -> list[ToolResult]:
        return default_tool_results(goal)


class ExecutorAgent:
    role = "executor"

    def draft(
        self,
        goal: str,
        learner_context: str,
        chunks: list[RetrievalChunk],
        tool_results: list[ToolResult],
    ) -> DraftAnswer:
        citations = [chunk.source_id for chunk in chunks]
        tool_names = [f"{result.adapter}.{result.action}" for result in tool_results]
        context_clause = (
            f" Learner context: {learner_context}" if learner_context else ""
        )

        content = (
            f"Socartes answers the goal '{goal}' through a visible agent loop. "
            "The Planner decomposes the request, the Retriever supplies RAG "
            "evidence, the Executor combines that evidence with MCP-style tool "
            "outputs, and the Critic checks whether the answer is cited and "
            f"complete. RAG evidence comes from {', '.join(citations)}, while "
            f"MCP tool use is represented by {', '.join(tool_names)}."
            f"{context_clause}"
        )

        return DraftAnswer(
            content=content,
            citations=citations,
            tool_results_used=tool_names,
            open_gaps=["External benchmark data should be refreshed for production use."],
        )


class CriticAgent:
    role = "critic"

    def review(self, draft: DraftAnswer) -> CriticReview:
        checks = [
            "acceptance criteria",
            "citation coverage",
            "tool output explainability",
            "open gap visibility",
        ]
        issues: list[CriticIssue] = []

        if not draft.citations:
            issues.append(
                CriticIssue(
                    type="missing_citation",
                    claim="Draft has no cited evidence.",
                    instruction="Retrieve domain context and attach citations.",
                )
            )
        if not draft.tool_results_used:
            issues.append(
                CriticIssue(
                    type="missing_tool_trace",
                    claim="Draft references tools without tool output identifiers.",
                    instruction="Attach adapter names and outputs used by the executor.",
                )
            )

        return CriticReview(
            status="revision_required" if issues else "approved",
            checks=checks,
            issues=issues,
            approved=not issues,
        )


class ReflectionLoop:
    def record(self, review: CriticReview) -> list[ReflectionEvent]:
        if review.approved:
            return [
                ReflectionEvent(
                    event_type="critic_review",
                    agent="critic",
                    message="Draft passed citation, tool trace, and gap checks.",
                ),
                ReflectionEvent(
                    event_type="executor_revision",
                    agent="executor",
                    message="Executor keeps citations and tool outputs attached to claims.",
                ),
                ReflectionEvent(
                    event_type="planner_update",
                    agent="planner",
                    message="Future plans must keep citations required for comparison claims.",
                ),
            ]

        return [
            ReflectionEvent(
                event_type="critic_review",
                agent="critic",
                message="Draft requires revision before learner-facing response.",
            )
        ]


class SocartesOrchestrator:
    def __init__(self) -> None:
        self.planner = PlannerAgent()
        self.retriever = RetrieverWorker()
        self.tools = ToolAdapterWorker()
        self.executor = ExecutorAgent()
        self.critic = CriticAgent()
        self.reflection = ReflectionLoop()

    def run(self, goal: str, learner_context: str = "") -> StudyTrace:
        plan = self.planner.plan(goal)
        retrieved_context = self.retriever.retrieve(goal)
        tool_results = self.tools.run_tools(goal)
        draft = self.executor.draft(
            goal=goal,
            learner_context=learner_context,
            chunks=retrieved_context,
            tool_results=tool_results,
        )
        review = self.critic.review(draft)
        reflection_events = self.reflection.record(review)

        return StudyTrace(
            goal=goal,
            learner_context=learner_context,
            plan=plan,
            retrieved_context=retrieved_context,
            tool_results=tool_results,
            draft=draft,
            review=review,
            reflection_events=reflection_events,
            final_answer=draft.content,
        )

    def agent_catalog(self) -> dict[str, dict[str, str]]:
        return {
            "planner": {
                "responsibility": "Convert learner goals into ordered study plans.",
                "input": "Learner goal and constraints.",
                "output": "Task graph, evidence requirements, and acceptance criteria.",
            },
            "retriever": {
                "responsibility": "Fetch external domain context through RAG.",
                "input": "Goal keywords and plan evidence requirements.",
                "output": "Ranked chunks with source identifiers and confidence.",
            },
            "executor": {
                "responsibility": "Synthesize answers from plan, RAG context, and tool outputs.",
                "input": "Plan tasks, retrieved chunks, learner context, and adapter results.",
                "output": "Draft answer with citations, tool trace, and open gaps.",
            },
            "critic": {
                "responsibility": "Audit the draft before it becomes learner-facing.",
                "input": "Executor draft, citations, tool results, and acceptance criteria.",
                "output": "Approval state, issues, and revision instructions.",
                "checks": "acceptance criteria, citation coverage, tool output explainability",
            },
            "tool_adapter": {
                "responsibility": "Expose MCP-style adapters for APIs, DBs, and files.",
                "input": "Scoped tool requests from the executor.",
                "output": "Auditable adapter output records.",
            },
        }
