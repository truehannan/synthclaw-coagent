"""
SynthClaw Agent Society -- Multi-Agent Orchestration System

Architecture:
- Single LLM, multiple personas (cost-efficient)
- Each "agent" is a role with its own system prompt injected per phase
- Orchestrator decomposes tasks -> delegates to specialist agents
- Shared context bus for inter-agent communication
- Agent lifecycle: spawn -> execute -> report -> archive/destroy
- Observer monitors execution, detects failures, validates final output

Agents are NOT separate processes. They are prompt-roles that the single
LLM assumes in sequence. The "society" is a structured execution pipeline
that gives better results than a monolithic prompt.
"""
from __future__ import annotations

import json
import re
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
    OBSERVING = "observing"
    WAITING = "waiting"
    COMPLETED = "completed"
    FAILED = "failed"


class AgentRole(Enum):
    ORCHESTRATOR = "orchestrator"
    PLANNER = "planner"
    EXECUTOR = "executor"
    RESEARCHER = "researcher"
    REVIEWER = "reviewer"
    OBSERVER = "observer"
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
#  AGENT REGISTRY -- Tracks all active agents in the society
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
        logger.info(f"[SOCIETY] Spawned: {name} ({role.value}) -- {task[:60]}")
        return agent

    def update(self, agent_id: str, status: AgentStatus, result: str = ""):
        """Update an agent's status."""
        if agent_id in self._agents:
            self._agents[agent_id].status = status
            if result:
                self._agents[agent_id].result = result
            if status in (AgentStatus.COMPLETED, AgentStatus.FAILED):
                self._agents[agent_id].completed_at = time.time()

    # Alias for backward compat (agent.py uses update_status in some places)
    def update_status(self, agent_id: str, status: AgentStatus, result: str = ""):
        return self.update(agent_id, status, result)

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

    def active(self) -> list[AgentInstance]:
        """All currently active agents."""
        return list(self._agents.values())

    # Alias
    def get_active(self) -> list[AgentInstance]:
        return self.active()

    def get_tree(self) -> list[dict]:
        """Get agent hierarchy as tree structure for CLI visualization."""
        roots = [a for a in self._agents.values() if not a.parent_id]
        def _tree(agent):
            node = agent.to_dict()
            node["children"] = [_tree(c) for c in agent.children if c.id in self._agents]
            return node
        return [_tree(r) for r in roots]

    def summary(self) -> dict:
        """Quick summary for status display."""
        active = self.active()
        return {
            "active": len(active),
            "by_status": {s.value: sum(1 for a in active if a.status == s) for s in AgentStatus if any(a.status == s for a in active)},
            "by_role": {r.value: sum(1 for a in active if a.role == r) for r in AgentRole if any(a.role == r for a in active)},
            "total_completed": len(self._history),
        }

    # Alias
    def get_summary(self) -> dict:
        return self.summary()

    def clear(self):
        """Reset all agents (new session)."""
        self._agents.clear()
        self._counter = 0


# Global registry instance
registry = AgentRegistry()


# ══════════════════════════════════════════════════════════════════════════════
#  ROLE PROMPTS -- System prompts injected per agent role
# ══════════════════════════════════════════════════════════════════════════════

ORCHESTRATOR_PROMPT = """\
You are the Orchestrator of SynthClaw Agent Society.

For SIMPLE requests (greetings, questions, quick tasks):
- Handle directly using tools. No delegation.

For COMPLEX requests (multi-step, research+action, deployments):
- Output a <plan> block to delegate:
<plan>
{"steps": [
  {"agent": "researcher", "task": "what to research"},
  {"agent": "executor", "task": "what to execute"},
  {"agent": "reviewer", "task": "validate results"}
]}
</plan>

Agents: researcher (search/gather info), executor (run commands/code), reviewer (validate), specialist (domain expert)
"""

PLANNER_PROMPT = """\
You are the Planner. Break down complex tasks into ordered steps.
Each step should be atomic and independently executable.
Output a numbered list of concrete actions. Be specific.
"""

EXECUTOR_PROMPT = """\
You are an Executor agent. Execute the assigned task immediately using tools.
Batch operations. No narration. Report what was done.

DANGEROUS OPERATIONS: If the task involves any of the following, respond ONLY with:
AWAITING_APPROVAL: <description of what you're about to do>

Dangerous patterns:
- rm -rf (recursive force delete)
- DROP TABLE / DROP DATABASE
- deploy to production
- billing / payment changes
- systemctl disable / stop critical services
- chmod 777 on system directories
- Writing to /etc/passwd, /etc/shadow, or similar

For all other operations, proceed normally.
"""

RESEARCHER_PROMPT = """\
You are a Researcher agent. Find information for the task.
Use web_search, google_search, scrape_page. Report findings with sources.
Do NOT execute commands -- only gather and report.
"""

REVIEWER_PROMPT = """\
You are the Reviewer. Validate completed work.
If correct: "APPROVED: [summary]"
If issues: "ISSUES: [problems and fixes needed]"
"""

OBSERVER_PROMPT = """\
You are the Observer agent. You monitor the execution of all other agents.

Your job:
1. Review the execution context (findings, results, reviews)
2. Detect failures, inconsistencies, or missed steps
3. Validate that the original objective was achieved
4. Check for security issues or unintended side effects

Output format:
- If everything is correct: "OBSERVATION: ALL CLEAR -- [brief summary of what succeeded]"
- If issues detected: "OBSERVATION: ISSUES DETECTED -- [list of problems]"
- If critical failure: "OBSERVATION: CRITICAL -- [what failed and recommended action]"

Be thorough but concise. Focus on correctness and completeness.
"""

SPECIALIST_PROMPT = """\
You are a Specialist agent for: {domain}. Handle this domain task with expertise.
"""


# ══════════════════════════════════════════════════════════════════════════════
#  SHARED CONTEXT BUS -- Inter-agent communication
# ══════════════════════════════════════════════════════════════════════════════

class SharedContext:
    """Shared memory/context between agents in a task session."""

    def __init__(self, objective: str = ""):
        self.objective = objective
        self.findings: list[dict] = []  # researcher outputs
        self.results: list[dict] = []   # executor outputs
        self.reviews: list[str] = []    # reviewer feedback
        self.observations: list[str] = []  # observer notes
        self.variables: dict[str, Any] = {}  # arbitrary shared state

    def add_finding(self, agent_id: str, content: str):
        self.findings.append({"agent": agent_id, "content": content, "ts": time.time()})

    def add_result(self, agent_id: str, content: str, ok: bool = True):
        self.results.append({"agent": agent_id, "content": content, "ok": ok, "ts": time.time()})

    def add_review(self, note: str):
        self.reviews.append(note)

    def add_observation(self, note: str):
        self.observations.append(note)

    def for_agent(self, role: str) -> str:
        """Build context string relevant to a specific agent role."""
        parts = []
        if self.objective:
            parts.append(f"OBJECTIVE: {self.objective}")
        if self.findings and role in ("executor", "reviewer", "observer", "orchestrator"):
            parts.append("RESEARCH:\n" + "\n".join(f"- {f['content'][:200]}" for f in self.findings[-5:]))
        if self.results and role in ("reviewer", "observer", "orchestrator"):
            parts.append("RESULTS:\n" + "\n".join(f"- {'OK' if r['ok'] else 'FAIL'}: {r['content'][:200]}" for r in self.results[-5:]))
        if self.reviews and role in ("observer", "orchestrator"):
            parts.append("REVIEWS:\n" + "\n".join(f"- {r[:200]}" for r in self.reviews[-3:]))
        if self.observations and role == "orchestrator":
            parts.append("OBSERVATIONS:\n" + "\n".join(f"- {o[:200]}" for o in self.observations[-3:]))
        return "\n\n".join(parts)

    def to_summary(self) -> str:
        return f"findings:{len(self.findings)} results:{len(self.results)} reviews:{len(self.reviews)} observations:{len(self.observations)}"


# ══════════════════════════════════════════════════════════════════════════════
#  ORCHESTRATION HELPERS
# ══════════════════════════════════════════════════════════════════════════════

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
    """Should this task use multi-agent delegation?

    Heuristic: short/simple messages handled directly, complex ones delegated.
    """
    if len(msg) < 40:
        return False
    indicators = [
        "and then", "after that", "first", "step by step",
        "deploy", "setup everything", "build a complete",
        "research and", "compare", "investigate",
        "create a full", "multi-step",
    ]
    msg_lower = msg.lower()
    if any(i in msg_lower for i in indicators):
        return True
    # Multi-sentence usually means complex
    if msg.count(".") >= 3 or msg.count(",") >= 4:
        return True
    return False


def get_role_prompt(role: AgentRole, context: SharedContext = None, domain: str = "") -> str:
    """Get the system prompt for a specific agent role, enriched with shared context."""
    prompts = {
        AgentRole.ORCHESTRATOR: ORCHESTRATOR_PROMPT,
        AgentRole.PLANNER: PLANNER_PROMPT,
        AgentRole.EXECUTOR: EXECUTOR_PROMPT,
        AgentRole.RESEARCHER: RESEARCHER_PROMPT,
        AgentRole.REVIEWER: REVIEWER_PROMPT,
        AgentRole.OBSERVER: OBSERVER_PROMPT,
        AgentRole.SPECIALIST: SPECIALIST_PROMPT.format(domain=domain or "general"),
    }
    prompt = prompts.get(role, EXECUTOR_PROMPT)
    if context:
        ctx = context.for_agent(role.value)
        if ctx:
            prompt += f"\n\n{ctx}"
    return prompt


# ══════════════════════════════════════════════════════════════════════════════
#  PUBLIC API -- Used by agent.py
# ══════════════════════════════════════════════════════════════════════════════

def get_society_status() -> dict:
    """Get current agent society state for CLI/Telegram display."""
    return {
        "agents": registry.summary(),
        "tree": registry.get_tree(),
        "active": [a.to_dict() for a in registry.active()],
    }


def reset_society():
    """Clear all agents (new conversation/session)."""
    registry.clear()
