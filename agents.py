"""
SynthClaw Agent Society — Multi-Agent Orchestration System

Architecture:
- Single LLM, multiple personas (cost-efficient)
- Each "agent" is a role with its own system prompt injected per phase
- Orchestrator decomposes tasks → delegates to specialist agents
- Shared context bus for inter-agent communication
- Agent lifecycle: spawn → execute → report → archive/destroy

Agents are NOT separate processes. They are prompt-roles that the single
LLM assumes in sequence. The "society" is a structured execution pipeline
that gives better results than a monolithic prompt.
"""
from __future__ import annotations

import json
import time
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
#  AGENT STATUS & TYPES
# ══════════════════════════════════════════════════════════════════════════════

class AgentStatus(Enum):
    IDLE = "idle"
    PLANNING = "planning"
    RESEARCHING = "researching"
    EXECUTING = "executing"
    REVIEWING = "reviewing"
    WAITING = "waiting"
    COMPLETED = "completed"
    FAILED = "failed"


class AgentRole(Enum):
    ORCHESTRATOR = "orchestrator"
    PLANNER = "planner"
    EXECUTOR = "executor"
    RESEARCHER = "researcher"
    REVIEWER = "reviewer"
    SPECIALIST = "specialist"  # dynamically spawned


@dataclass
class AgentInstance:
    """A running agent in the society."""
    id: str
    role: AgentRole
    name: str
    status: AgentStatus = AgentStatus.IDLE
    task: str = ""
    result: str = ""
    parent_id: str = ""
    children: list = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    completed_at: float = 0.0
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id, "role": self.role.value, "name": self.name,
            "status": self.status.value, "task": self.task[:100],
            "children": [c.id for c in self.children],
            "elapsed": round(time.time() - self.created_at, 1),
        }


# ══════════════════════════════════════════════════════════════════════════════
#  AGENT REGISTRY — Tracks all active agents in the society
# ══════════════════════════════════════════════════════════════════════════════

class AgentRegistry:
    """Central registry of all active agent instances."""

    def __init__(self):
        self._agents: dict[str, AgentInstance] = {}
        self._counter: int = 0
        self._history: list[dict] = []  # completed agent summaries

    def spawn(self, role: AgentRole, name: str, task: str = "", parent_id: str = "") -> AgentInstance:
        """Create and register a new agent."""
        self._counter += 1
        agent_id = f"{role.value}_{self._counter}"
        agent = AgentInstance(
            id=agent_id, role=role, name=name,
            task=task, parent_id=parent_id,
        )
        self._agents[agent_id] = agent
        # Link to parent
        if parent_id and parent_id in self._agents:
            self._agents[parent_id].children.append(agent)
        logger.info(f"Agent spawned: {name} ({role.value}) — {task[:60]}")
        return agent

    def update_status(self, agent_id: str, status: AgentStatus, result: str = ""):
        """Update an agent's status."""
        if agent_id in self._agents:
            self._agents[agent_id].status = status
            if result:
                self._agents[agent_id].result = result
            if status in (AgentStatus.COMPLETED, AgentStatus.FAILED):
                self._agents[agent_id].completed_at = time.time()

    def destroy(self, agent_id: str):
        """Archive and remove an agent."""
        if agent_id in self._agents:
            agent = self._agents.pop(agent_id)
            self._history.append(agent.to_dict())
            # Remove from parent's children
            if agent.parent_id and agent.parent_id in self._agents:
                parent = self._agents[agent.parent_id]
                parent.children = [c for c in parent.children if c.id != agent_id]

    def get(self, agent_id: str) -> Optional[AgentInstance]:
        return self._agents.get(agent_id)

    def get_active(self) -> list[AgentInstance]:
        """All currently active agents."""
        return list(self._agents.values())

    def get_tree(self) -> list[dict]:
        """Get agent hierarchy as tree structure for CLI visualization."""
        roots = [a for a in self._agents.values() if not a.parent_id]
        def build_tree(agent):
            node = agent.to_dict()
            node["children"] = [build_tree(c) for c in agent.children if c.id in self._agents]
            return node
        return [build_tree(r) for r in roots]

    def get_summary(self) -> dict:
        """Quick summary for status display."""
        active = self.get_active()
        return {
            "total_active": len(active),
            "by_status": {s.value: sum(1 for a in active if a.status == s) for s in AgentStatus if any(a.status == s for a in active)},
            "by_role": {r.value: sum(1 for a in active if a.role == r) for r in AgentRole if any(a.role == r for a in active)},
            "completed_total": len(self._history),
        }

    def clear(self):
        """Reset all agents (new session)."""
        self._agents.clear()
        self._counter = 0


# Global registry instance
registry = AgentRegistry()


# ══════════════════════════════════════════════════════════════════════════════
#  AGENT PROMPTS — Role-specific system prompts injected per phase
# ══════════════════════════════════════════════════════════════════════════════

ORCHESTRATOR_PROMPT = """\
You are the Orchestrator of SynthClaw Agent Society — a multi-agent system.

Your job: Analyze the user's request and decide the execution strategy.

For SIMPLE requests (greetings, questions, quick tasks):
- Handle directly. No delegation needed. Just respond or use tools.

For COMPLEX requests (multi-step tasks, deployments, research + action):
- Output a PLAN as JSON with this structure:
<plan>
{{"steps": [
  {{"agent": "researcher", "task": "description of what to research"}},
  {{"agent": "executor", "task": "description of what to execute"}},
  {{"agent": "reviewer", "task": "verify the results"}}
]}}
</plan>

Available agents: researcher, executor, reviewer, specialist
- researcher: web search, docs lookup, information gathering
- executor: shell commands, file operations, code execution, API calls
- reviewer: validates results, checks for errors
- specialist: spawned for domain-specific tasks (specify domain in task)

If the request is simple enough to handle directly, just use tools normally.
Only output a <plan> if the task genuinely benefits from decomposition.
"""

PLANNER_PROMPT = """\
You are the Planner. Break down complex tasks into ordered steps.
Each step should be atomic and independently executable.
Output a numbered list of concrete actions. Be specific.
"""

EXECUTOR_PROMPT = """\
You are the Executor agent. You receive a specific task and execute it immediately.
Use tools aggressively. Batch operations. No narration — just act.
Report: what was done, what succeeded, what failed.
"""

RESEARCHER_PROMPT = """\
You are the Researcher agent. Gather information needed for the task.
Use web_search, google_search, scrape_page to find answers.
Report findings concisely. Include URLs for sources.
Do NOT execute — only research and report facts.
"""

REVIEWER_PROMPT = """\
You are the Reviewer agent. Validate completed work.
Check: Did the execution achieve the objective? Any errors? Missing pieces?
If everything is correct, respond: "APPROVED: [summary]"
If issues found, respond: "ISSUES: [what's wrong and how to fix]"
"""

SPECIALIST_PROMPT = """\
You are a Specialist agent spawned for: {domain}
Handle this specific domain task with expertise.
Use relevant tools. Be thorough but concise.
"""


# ══════════════════════════════════════════════════════════════════════════════
#  SHARED CONTEXT BUS — Inter-agent communication
# ══════════════════════════════════════════════════════════════════════════════

class SharedContext:
    """Shared memory/context between agents in a task session."""

    def __init__(self):
        self.task_objective: str = ""
        self.findings: list[dict] = []  # researcher outputs
        self.execution_results: list[dict] = []  # executor outputs
        self.review_notes: list[str] = []  # reviewer feedback
        self.variables: dict[str, Any] = {}  # arbitrary shared state
        self.messages_log: list[dict] = []  # inter-agent messages

    def add_finding(self, agent_id: str, content: str):
        self.findings.append({"agent": agent_id, "content": content, "ts": time.time()})

    def add_result(self, agent_id: str, content: str, success: bool = True):
        self.execution_results.append({"agent": agent_id, "content": content, "success": success, "ts": time.time()})

    def add_review(self, note: str):
        self.review_notes.append(note)

    def send_message(self, from_agent: str, to_agent: str, content: str):
        self.messages_log.append({"from": from_agent, "to": to_agent, "content": content, "ts": time.time()})

    def get_context_for_agent(self, agent_role: str) -> str:
        """Build context string relevant to a specific agent role."""
        parts = []
        if self.task_objective:
            parts.append(f"OBJECTIVE: {self.task_objective}")
        if self.findings and agent_role in ("executor", "reviewer", "orchestrator"):
            parts.append("RESEARCH FINDINGS:\n" + "\n".join(f"- {f['content'][:200]}" for f in self.findings[-5:]))
        if self.execution_results and agent_role in ("reviewer", "orchestrator"):
            parts.append("EXECUTION RESULTS:\n" + "\n".join(f"- {'✓' if r['success'] else '✗'} {r['content'][:200]}" for r in self.execution_results[-5:]))
        if self.review_notes and agent_role == "orchestrator":
            parts.append("REVIEW NOTES:\n" + "\n".join(f"- {n[:200]}" for n in self.review_notes[-3:]))
        return "\n\n".join(parts) if parts else ""

    def to_summary(self) -> str:
        """One-line summary for status display."""
        return f"findings:{len(self.findings)} results:{len(self.execution_results)} reviews:{len(self.review_notes)}"


# ══════════════════════════════════════════════════════════════════════════════
#  ORCHESTRATOR — Decides whether to delegate or handle directly
# ══════════════════════════════════════════════════════════════════════════════

import re

def parse_plan(response: str) -> list[dict] | None:
    """Extract <plan>...</plan> JSON from orchestrator response."""
    match = re.search(r"<plan>\s*(\{[\s\S]*?\})\s*</plan>", response)
    if not match:
        return None
    try:
        data = json.loads(match.group(1))
        steps = data.get("steps", [])
        if isinstance(steps, list) and all(isinstance(s, dict) for s in steps):
            return steps
    except (json.JSONDecodeError, KeyError):
        pass
    return None


def get_agent_prompt(role: AgentRole, context: SharedContext, domain: str = "") -> str:
    """Get the system prompt for a specific agent role, enriched with shared context."""
    base_prompts = {
        AgentRole.ORCHESTRATOR: ORCHESTRATOR_PROMPT,
        AgentRole.PLANNER: PLANNER_PROMPT,
        AgentRole.EXECUTOR: EXECUTOR_PROMPT,
        AgentRole.RESEARCHER: RESEARCHER_PROMPT,
        AgentRole.REVIEWER: REVIEWER_PROMPT,
        AgentRole.SPECIALIST: SPECIALIST_PROMPT.format(domain=domain or "general"),
    }
    prompt = base_prompts.get(role, EXECUTOR_PROMPT)
    ctx = context.get_context_for_agent(role.value)
    if ctx:
        prompt += f"\n\nCURRENT CONTEXT:\n{ctx}"
    return prompt


def should_delegate(user_message: str) -> bool:
    """Heuristic: should this task be decomposed into multi-agent steps?
    Simple messages get handled directly. Complex tasks get delegated.
    """
    msg = user_message.lower()
    # Short messages are usually simple
    if len(msg) < 50:
        return False
    # Explicit complexity indicators
    complex_indicators = [
        "and then", "after that", "first", "second", "third",
        "deploy", "setup", "configure", "build and",
        "research", "investigate", "compare",
        "create a full", "build a complete",
        "step by step", "multi-step",
    ]
    if any(ind in msg for ind in complex_indicators):
        return True
    # Multi-sentence usually means complex
    if msg.count(".") >= 3 or msg.count(",") >= 4:
        return True
    return False


# ══════════════════════════════════════════════════════════════════════════════
#  PUBLIC API — Used by agent.py
# ══════════════════════════════════════════════════════════════════════════════

def get_society_status() -> dict:
    """Get current agent society state for CLI/Telegram display."""
    return {
        "agents": registry.get_summary(),
        "tree": registry.get_tree(),
        "active": [a.to_dict() for a in registry.get_active()],
    }


def reset_society():
    """Clear all agents (new conversation/session)."""
    registry.clear()



# ══════════════════════════════════════════════════════════════════════════════
#  AGENT STATUS & TYPES
# ══════════════════════════════════════════════════════════════════════════════

class AgentStatus(Enum):
    IDLE = "idle"
    PLANNING = "planning"
    RESEARCHING = "researching"
    EXECUTING = "executing"
    REVIEWING = "reviewing"
    WAITING = "waiting"
    COMPLETED = "completed"
    FAILED = "failed"


class AgentRole(Enum):
    ORCHESTRATOR = "orchestrator"
    PLANNER = "planner"
    EXECUTOR = "executor"
    RESEARCHER = "researcher"
    REVIEWER = "reviewer"
    SPECIALIST = "specialist"


@dataclass
class AgentInstance:
    id: str
    role: AgentRole
    name: str
    status: AgentStatus = AgentStatus.IDLE
    task: str = ""
    result: str = ""
    parent_id: str = ""
    children: list = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    completed_at: float = 0.0

    def to_dict(self) -> dict:
        return {
            "id": self.id, "role": self.role.value, "name": self.name,
            "status": self.status.value, "task": self.task[:100],
            "children": [c.id for c in self.children],
            "elapsed": round(time.time() - self.created_at, 1),
        }



# ══════════════════════════════════════════════════════════════════════════════
#  AGENT REGISTRY
# ══════════════════════════════════════════════════════════════════════════════

class AgentRegistry:
    def __init__(self):
        self._agents: dict[str, AgentInstance] = {}
        self._counter: int = 0
        self._history: list[dict] = []

    def spawn(self, role: AgentRole, name: str, task: str = "", parent_id: str = "") -> AgentInstance:
        self._counter += 1
        agent_id = f"{role.value}_{self._counter}"
        agent = AgentInstance(id=agent_id, role=role, name=name, task=task, parent_id=parent_id)
        self._agents[agent_id] = agent
        if parent_id and parent_id in self._agents:
            self._agents[parent_id].children.append(agent)
        logger.info(f"[SOCIETY] Spawned: {name} ({role.value})")
        return agent

    def update(self, agent_id: str, status: AgentStatus, result: str = ""):
        if agent_id in self._agents:
            self._agents[agent_id].status = status
            if result:
                self._agents[agent_id].result = result
            if status in (AgentStatus.COMPLETED, AgentStatus.FAILED):
                self._agents[agent_id].completed_at = time.time()

    def destroy(self, agent_id: str):
        if agent_id in self._agents:
            agent = self._agents.pop(agent_id)
            self._history.append(agent.to_dict())

    def get(self, agent_id: str) -> Optional[AgentInstance]:
        return self._agents.get(agent_id)

    def active(self) -> list[AgentInstance]:
        return list(self._agents.values())

    def get_tree(self) -> list[dict]:
        roots = [a for a in self._agents.values() if not a.parent_id]
        def _tree(a):
            d = a.to_dict()
            d["children"] = [_tree(c) for c in a.children if c.id in self._agents]
            return d
        return [_tree(r) for r in roots]

    def summary(self) -> dict:
        active = self.active()
        return {
            "active": len(active),
            "by_role": {r.value: sum(1 for a in active if a.role == r) for r in AgentRole if any(a.role == r for a in active)},
            "total_completed": len(self._history),
        }

    def clear(self):
        self._agents.clear()
        self._counter = 0

registry = AgentRegistry()



# ══════════════════════════════════════════════════════════════════════════════
#  SHARED CONTEXT BUS
# ══════════════════════════════════════════════════════════════════════════════

class SharedContext:
    def __init__(self, objective: str = ""):
        self.objective = objective
        self.findings: list[dict] = []
        self.results: list[dict] = []
        self.reviews: list[str] = []

    def add_finding(self, agent_id: str, content: str):
        self.findings.append({"agent": agent_id, "content": content, "ts": time.time()})

    def add_result(self, agent_id: str, content: str, ok: bool = True):
        self.results.append({"agent": agent_id, "content": content, "ok": ok, "ts": time.time()})

    def add_review(self, note: str):
        self.reviews.append(note)

    def for_agent(self, role: str) -> str:
        parts = []
        if self.objective:
            parts.append(f"OBJECTIVE: {self.objective}")
        if self.findings and role in ("executor", "reviewer", "orchestrator"):
            parts.append("RESEARCH:\n" + "\n".join(f"- {f['content'][:200]}" for f in self.findings[-5:]))
        if self.results and role in ("reviewer", "orchestrator"):
            parts.append("RESULTS:\n" + "\n".join(f"- {'OK' if r['ok'] else 'FAIL'}: {r['content'][:200]}" for r in self.results[-5:]))
        if self.reviews and role == "orchestrator":
            parts.append("REVIEWS:\n" + "\n".join(f"- {r[:200]}" for r in self.reviews[-3:]))
        return "\n\n".join(parts)



# ══════════════════════════════════════════════════════════════════════════════
#  ROLE PROMPTS
# ══════════════════════════════════════════════════════════════════════════════

ORCHESTRATOR_PROMPT = """\
You are the Orchestrator of SynthClaw Agent Society.

For SIMPLE requests (greetings, questions, quick tasks):
- Handle directly using tools. No delegation.

For COMPLEX requests (multi-step, research+action, deployments):
- Output a <plan> block to delegate:
<plan>
{{"steps": [
  {{"agent": "researcher", "task": "what to research"}},
  {{"agent": "executor", "task": "what to execute"}},
  {{"agent": "reviewer", "task": "validate results"}}
]}}
</plan>

Agents: researcher (search/gather info), executor (run commands/code), reviewer (validate), specialist (domain expert)
"""

EXECUTOR_PROMPT = """\
You are an Executor agent. Execute the assigned task immediately using tools.
Batch operations. No narration. Report what was done.
"""

RESEARCHER_PROMPT = """\
You are a Researcher agent. Find information for the task.
Use web_search, google_search, scrape_page. Report findings with sources.
Do NOT execute commands — only gather and report.
"""

REVIEWER_PROMPT = """\
You are the Reviewer. Validate completed work.
If correct: "APPROVED: [summary]"
If issues: "ISSUES: [problems and fixes needed]"
"""

SPECIALIST_PROMPT = """\
You are a Specialist agent for: {domain}. Handle this domain task with expertise.
"""


def get_role_prompt(role: AgentRole, context: SharedContext = None, domain: str = "") -> str:
    prompts = {
        AgentRole.ORCHESTRATOR: ORCHESTRATOR_PROMPT,
        AgentRole.EXECUTOR: EXECUTOR_PROMPT,
        AgentRole.RESEARCHER: RESEARCHER_PROMPT,
        AgentRole.REVIEWER: REVIEWER_PROMPT,
        AgentRole.SPECIALIST: SPECIALIST_PROMPT.format(domain=domain or "general"),
    }
    prompt = prompts.get(role, EXECUTOR_PROMPT)
    if context:
        ctx = context.for_agent(role.value)
        if ctx:
            prompt += f"\n\n{ctx}"
    return prompt



# ══════════════════════════════════════════════════════════════════════════════
#  ORCHESTRATION LOGIC
# ══════════════════════════════════════════════════════════════════════════════

import re

def parse_plan(text: str) -> list[dict] | None:
    """Extract <plan>JSON</plan> from orchestrator output."""
    m = re.search(r"<plan>\s*(\{[\s\S]*?\})\s*</plan>", text)
    if not m:
        return None
    try:
        data = json.loads(m.group(1))
        steps = data.get("steps", [])
        if isinstance(steps, list) and steps:
            return steps
    except (json.JSONDecodeError, KeyError):
        pass
    return None


def should_delegate(msg: str) -> bool:
    """Should this task use multi-agent delegation?"""
    if len(msg) < 40:
        return False
    indicators = ["and then", "after that", "first", "step by step", "deploy", "setup everything", "build a complete", "research and", "compare"]
    return any(i in msg.lower() for i in indicators) or msg.count(".") >= 3


# ══════════════════════════════════════════════════════════════════════════════
#  PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

def get_society_status() -> dict:
    return {"agents": registry.summary(), "tree": registry.get_tree(), "active": [a.to_dict() for a in registry.active()]}

def reset_society():
    registry.clear()
