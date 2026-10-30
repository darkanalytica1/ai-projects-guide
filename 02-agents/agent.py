"""Long-horizon agent: plan -> parallel sub-agents -> writer, with guardrails.

  python agent.py "Research what HNSW is and write report.md with a small Python benchmark"
"""
import asyncio
import os
import pathlib
import re
import sys
from decimal import Decimal

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_ai import (Agent, ApprovalRequired, ModelRetry, DeferredToolRequests, DeferredToolResults, RunContext,
                         Tool, ToolDenied, UsageLimitExceeded, UsageLimits)
from pydantic_ai.capabilities import HandleDeferredToolCalls
from pydantic_ai.messages import ModelResponse, ToolCallPart, ToolReturnPart

from logger import StepLogger
from sandbox import run_code

load_dotenv()
os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")
WORKSPACE = pathlib.Path("workspace").resolve()
WORKSPACE.mkdir(exist_ok=True)
MAX_STEPS = int(os.getenv("MAX_STEPS", "25"))
MAX_COST = Decimal(os.getenv("MAX_COST_USD", "0.50"))
ALLOWED_TOOLS = {"search_files", "read_file", "write_file", "run_python"}  # allowlist


def get_model():
    name = os.getenv("AGENT_MODEL", "openai/gpt-4o-mini")
    if name == "test":                       # offline dry run, no API key, no cost
        from pydantic_ai.models.test import TestModel
        return TestModel()
    from pydantic_ai.models.openrouter import OpenRouterModel
    return OpenRouterModel(name)             # reads OPENROUTER_API_KEY


# ---------- typed tools (Pydantic validates every argument) ----------
def _safe(path: str) -> pathlib.Path:
    p = (WORKSPACE / path).resolve()
    if WORKSPACE not in p.parents and p != WORKSPACE:
        raise ValueError("path escapes the workspace")   # guardrail: no ../../etc
    return p


def search_files(query: str) -> list[str]:
    """Search the workspace for files whose text contains the query (case-insensitive)."""
    hits = []
    for f in WORKSPACE.rglob("*"):
        if f.is_file() and query.lower() in f.read_text(errors="ignore").lower():
            hits.append(str(f.relative_to(WORKSPACE)))
    return hits[:20] or ["no matches"]


def read_file(path: str) -> str:
    """Read a text file from the workspace (max 8000 characters)."""
    p = _safe(path)
    if not p.is_file():
        raise ModelRetry(f"{path} does not exist; call search_files first")  # model retries
    return p.read_text(errors="ignore")[:8000]


def write_file(ctx: RunContext, path: str, content: str) -> str:
    """Create or overwrite a text file in the workspace."""
    p = _safe(path)
    if p.exists() and not ctx.tool_call_approved:
        raise ApprovalRequired          # overwriting = risky -> ask a human first
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    return f"wrote {len(content)} chars to {path}"


def run_python(code: str) -> str:
    """Execute Python code in an isolated sandbox (E2B or Docker) and return stdout/stderr."""
    if re.search(r"\b(rm -rf|shutil\.rmtree|os\.remove)\b", code):
        return "blocked: destructive operation"          # cheap static guardrail
    try:
        return run_code(code)
    except Exception as e:                               # report, don't crash the run
        return f"sandbox error: {type(e).__name__}: {e}"


def _tools():
    all_tools = [Tool(search_files), Tool(read_file), Tool(write_file), Tool(run_python)]
    return [t for t in all_tools if t.name in ALLOWED_TOOLS]


async def ask_human(ctx: RunContext, requests: DeferredToolRequests) -> DeferredToolResults:
    approvals = {}
    for call in requests.approvals:
        if os.getenv("AUTO_APPROVE") == "1":
            approvals[call.tool_call_id] = True
            continue
        ans = await asyncio.to_thread(input, f"\nApprove {call.tool_name}({call.args})? [y/N] ")
        approvals[call.tool_call_id] = True if ans.strip().lower() == "y" else ToolDenied("Human said no")
    return requests.build_results(approvals=approvals)


# ---------- agents ----------
class Plan(BaseModel):
    subtasks: list[str] = Field(min_length=1, max_length=4,
                                description="independent subtasks that can run in parallel")


planner = Agent(get_model(), output_type=Plan,
                instructions="Split the task into 1-4 independent subtasks. Do not do them.")

worker = Agent(get_model(), tools=_tools(),
               capabilities=[HandleDeferredToolCalls(handler=ask_human)],
               instructions=("You complete ONE subtask using tools. Save findings to "
                             "workspace files named notes_<topic>.md. Use run_python to verify "
                             "any code. Finish with a 3-line summary."))

writer = Agent(get_model(), tools=_tools(),
               capabilities=[HandleDeferredToolCalls(handler=ask_human)],
               instructions="Read the notes_*.md files and write the final deliverable requested.")


async def logged_run(agent: Agent, prompt: str, log: StepLogger, name: str):
    """agent.iter() lets us see every model response and tool result as it happens."""
    limits = UsageLimits(request_limit=MAX_STEPS, tool_calls_limit=MAX_STEPS * 2,
                         cost_limit=MAX_COST)
    async with agent.iter(prompt, usage_limits=limits) as run:
        async for node in run:
            if Agent.is_call_tools_node(node):
                resp: ModelResponse = node.model_response
                u = resp.usage
                log.log(kind="model", agent=name, model=resp.model_name,
                        input_tokens=u.input_tokens, output_tokens=u.output_tokens,
                        cost_usd=getattr(u, "cost", None))
                for part in resp.parts:
                    if isinstance(part, ToolCallPart):
                        log.log(kind="call", agent=name, tool=part.tool_name, input=part.args)
            elif Agent.is_model_request_node(node):
                for part in node.request.parts:
                    if isinstance(part, ToolReturnPart):
                        log.log(kind="result", agent=name, tool=part.tool_name,
                                output=str(part.content)[:500])
        return run.result.output, run.usage


async def main(task: str):
    log = StepLogger("run")
    plan, _ = await logged_run(planner, task, log, "planner")
    print("plan:", plan.subtasks)
    # sub-agents in parallel; one failure doesn't kill the others
    results = await asyncio.gather(
        *[logged_run(worker, f"Overall task: {task}\nYour subtask: {s}", log, f"worker{i}")
          for i, s in enumerate(plan.subtasks)],
        return_exceptions=True)
    for s, r in zip(plan.subtasks, results):
        status = "ERROR " + repr(r) if isinstance(r, Exception) else "ok"
        print(f"- {s[:60]}: {status}")
    final, usage = await logged_run(writer, f"Task: {task}", log, "writer")
    print("\nFINAL:\n", final)
    print(f"\nlog file: {log.path}")


if __name__ == "__main__":
    try:
        asyncio.run(main(" ".join(sys.argv[1:]) or "Write hello.md that says hello"))
    except UsageLimitExceeded as e:
        print("STOPPED by budget/step guardrail:", e)
