const roles = {
  planner: {
    badge: "P",
    role: "Planner agent",
    title: "Breaks the goal into evidence-backed tasks.",
    description:
      "The planner defines the research route, assigns retrieval targets, and creates acceptance criteria before execution.",
    tasks: [
      "Define the learner question and expected output.",
      "Identify external knowledge sources for retrieval.",
      "Route grounded subtasks to the executor.",
    ],
    score: 86,
  },
  executor: {
    badge: "E",
    role: "Executor agent",
    title: "Synthesizes the answer with retrieved context and tools.",
    description:
      "The executor combines RAG snippets, tool outputs, and the learning goal into a traceable draft.",
    tasks: [
      "Read the retrieved evidence and filter weak claims.",
      "Call tools only when external state is required.",
      "Draft an answer that keeps citations attached to claims.",
    ],
    score: 91,
  },
  critic: {
    badge: "C",
    role: "Critic agent",
    title: "Audits claims, missing evidence, and revision quality.",
    description:
      "The critic checks whether the response follows the goal, cites evidence, and records revision requests.",
    tasks: [
      "Challenge unsupported comparisons.",
      "Mark unclear tool outputs for rerun or removal.",
      "Approve the answer only after the reflection loop is complete.",
    ],
    score: 78,
  },
};

const sources = {
  paper: {
    type: "Domain note",
    title: "Agent workflow brief",
    body:
      "Multi-agent systems separate planning, execution, and critique to make reasoning steps inspectable and easier to revise.",
    citation: "citation: workflow-note-01",
    confidence: "confidence: high",
  },
  db: {
    type: "Vector DB",
    title: "Learning systems index",
    body:
      "Retrieval-augmented generation grounds answers in external references before generation, reducing unsupported claims in study workflows.",
    citation: "citation: rag-index-18",
    confidence: "confidence: medium",
  },
  files: {
    type: "Files",
    title: "Course artifact folder",
    body:
      "Local artifacts provide learner-specific context such as notes, drafts, rubrics, and previous revisions.",
    citation: "citation: local-files-07",
    confidence: "confidence: high",
  },
};

const tools = {
  api: `adapter: external_api
status: ready
result: domain data can be attached to the execution trace`,
  database: `adapter: knowledge_database
status: ready
query: multi-agent learning workflow
result: 12 ranked notes returned with citation metadata`,
  files: `adapter: filesystem
status: ready
scope: learner_workspace
result: 4 study artifacts available for grounded revision`,
};

const cycleSteps = [
  {
    score: 82,
    timeline: [
      "Planner creates retrieval targets for the learner question.",
      "Executor drafts a comparison from two evidence snippets.",
      "Critic queues a citation check before final response.",
    ],
  },
  {
    score: 89,
    timeline: [
      "Critic flags a missing citation for the comparison claim.",
      "Executor retrieves a stronger source and rewrites the claim.",
      "Planner updates the next task with an evidence requirement.",
    ],
  },
  {
    score: 94,
    timeline: [
      "Executor reruns the database adapter with a narrower query.",
      "Critic approves the revised claim and records residual risk.",
      "Reflection loop stores the pattern for the next learning task.",
    ],
  },
];

let cycleIndex = 0;

const roleButtons = document.querySelectorAll(".role-button");
const sourceButtons = document.querySelectorAll(".source-item");
const toolButtons = document.querySelectorAll(".tool-card");
const taskList = document.querySelector("#taskList");
const groundingScore = document.querySelector("#groundingScore");
const groundingBar = document.querySelector("#groundingBar");
const timeline = document.querySelector("#reflectionTimeline");

function setGrounding(score) {
  groundingScore.textContent = `${score}%`;
  groundingBar.style.width = `${score}%`;
}

function setRole(roleKey) {
  const role = roles[roleKey];

  document.querySelector("#agentBadge").textContent = role.badge;
  document.querySelector("#agentRole").textContent = role.role;
  document.querySelector("#agentTitle").textContent = role.title;
  document.querySelector("#agentDescription").textContent = role.description;
  taskList.innerHTML = role.tasks.map((task) => `<li>${task}</li>`).join("");
  setGrounding(role.score);

  roleButtons.forEach((button) => {
    button.classList.toggle("active", button.dataset.role === roleKey);
  });
}

function setSource(sourceKey) {
  const source = sources[sourceKey];

  document.querySelector("#sourceType").textContent = source.type;
  document.querySelector("#sourceTitle").textContent = source.title;
  document.querySelector("#sourceBody").textContent = source.body;
  document.querySelector("#sourceCitation").textContent = source.citation;
  document.querySelector("#sourceConfidence").textContent = source.confidence;

  sourceButtons.forEach((button) => {
    button.classList.toggle("active", button.dataset.source === sourceKey);
  });
}

function setTool(toolKey) {
  document.querySelector("#toolOutput").textContent = tools[toolKey];

  toolButtons.forEach((button) => {
    button.classList.toggle("active", button.dataset.tool === toolKey);
  });
}

function runReflectionCycle() {
  const next = cycleSteps[cycleIndex % cycleSteps.length];
  cycleIndex += 1;
  setGrounding(next.score);

  timeline.innerHTML = next.timeline
    .map(
      (item, index) => `
        <div class="timeline-item ${index === 0 ? "current" : ""}">
          <span>${index + 1}</span>
          <p>${item}</p>
        </div>
      `,
    )
    .join("");
}

roleButtons.forEach((button) => {
  button.addEventListener("click", () => setRole(button.dataset.role));
});

sourceButtons.forEach((button) => {
  button.addEventListener("click", () => setSource(button.dataset.source));
});

toolButtons.forEach((button) => {
  button.addEventListener("click", () => setTool(button.dataset.tool));
});

document
  .querySelector("#runCycleButton")
  .addEventListener("click", runReflectionCycle);
