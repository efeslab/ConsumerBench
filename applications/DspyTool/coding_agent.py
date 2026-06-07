"""A long-running, multi-step coding agent that simulates an OpenClaw/SERA-style
local coding assistant occupying the GPU for sustained, stateful work.

It plugs into ConsumerBench's generic ``DspyTool`` application via a ``tool_string``
pointing at :class:`CodingAgent`. Each ``forward()`` call runs a multi-turn coding
session against the shared llama.cpp OpenAI-compatible server: the agent is asked to
build a small software project step by step (scaffold -> implement -> add tests ->
fix bugs -> refactor -> document ...). Crucially, the *entire growing conversation*
(all prior turns) is resent each step, so the prompt — and therefore the server-side
KV-cache footprint — grows monotonically, modeling an agent with expensive-to-preempt
KV state.

This is the "sustained, long-running agent" half of the asymmetric scheduling
scenario from the rebuttal: it is colocated with a bursty, latency-sensitive Chatbot
on the same shared model, and we measure how the agent's end-to-end latency inflates
relative to running in isolation (preemption cost) — and whether it meets its SLO.
"""

import json
import os
import time
import urllib.request


# A sequence of coding-task instructions. Each turn appends to the conversation,
# so context grows step over step (KV-cache pressure). The tasks are deliberately
# generic so the agent keeps producing code regardless of the base model.
DEFAULT_STEPS = [
    "You are a senior software engineer building a small Python library called `taskq`, "
    "an in-memory priority task queue. Start by writing the core `TaskQueue` class with "
    "`push(item, priority)` and `pop()` methods. Output only code.",
    "Add a `peek()` method and a `__len__` to the TaskQueue class. Show the full updated class.",
    "Now add thread-safety to TaskQueue using a lock. Show the full updated class.",
    "Write a pytest test suite covering push/pop ordering, peek, and concurrent access.",
    "There is a subtle bug: equal-priority items should be FIFO. Fix the class to guarantee "
    "FIFO order among equal priorities, and explain the fix.",
    "Add a `pop_batch(n)` method that pops up to n highest-priority items efficiently. Show code.",
    "Refactor the implementation to use Python's heapq for O(log n) push/pop while keeping "
    "FIFO tie-breaking. Show the full refactored class.",
    "Add type hints and docstrings throughout, and add a `to_list()` method for debugging.",
    "Write a short benchmark script that pushes 100k items and times pop_batch. Show code.",
    "Summarize the final design, list the public API, and note remaining limitations.",
]


def _wait_for_server(api_base, timeout=180):
    """Block until the OpenAI-compatible server answers, or raise on timeout."""
    url = f"{api_base.rstrip('/')}/v1/models"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            time.sleep(2)
    raise RuntimeError(f"Server at {api_base} not ready after {timeout}s")


def _post_chat(api_base, model, messages, max_tokens, timeout=600):
    """One blocking chat-completion call to the local OpenAI-compatible server."""
    url = f"{api_base.rstrip('/')}/v1/chat/completions"
    payload = json.dumps({
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": False,
    }).encode("utf-8")
    req = urllib.request.Request(url, data=payload,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    msg = data["choices"][0]["message"]
    # llama.cpp puts reasoning-model output in reasoning_content when thinking is on;
    # with thinking disabled the answer is in content.
    content = msg.get("content") or msg.get("reasoning_content") or ""
    usage = data.get("usage", {})
    return content, usage


class CodingAgent:
    """Multi-step coding agent. Configure via DspyTool tool_kwargs.

    tool_kwargs:
        api_port (int):     llama.cpp server port (default 8080)
        model (str):        model id sent to the server (default 'Qwen3-8B')
        num_steps (int):    number of sequential coding turns (default 10)
        max_tokens (int):   max tokens per turn (default 512)
        slo_seconds (float):end-to-end latency SLO for the whole session (default None)
        results_dir (str):  where to write a per-step latency log (optional)
    """

    def forward(self, api_port=8080, model="Qwen3-8B", num_steps=10, max_tokens=512,
                slo_seconds=None, results_dir=None, **_ignored):
        api_base = f"http://127.0.0.1:{api_port}"
        _wait_for_server(api_base)
        steps = DEFAULT_STEPS[:num_steps] if num_steps <= len(DEFAULT_STEPS) else \
            (DEFAULT_STEPS * ((num_steps // len(DEFAULT_STEPS)) + 1))[:num_steps]

        messages = [{"role": "system",
                     "content": "You are a careful, expert software engineer. "
                                "Always return complete, runnable code."}]
        step_latencies = []
        total_prompt_tokens = 0
        total_completion_tokens = 0

        session_start = time.time()
        for i, instruction in enumerate(steps):
            messages.append({"role": "user", "content": instruction})
            t0 = time.time()
            content, usage = _post_chat(api_base, model, messages, max_tokens)
            dt = time.time() - t0
            step_latencies.append(dt)
            # Append the assistant turn so context (and KV footprint) keeps growing.
            messages.append({"role": "assistant", "content": content})
            total_prompt_tokens += usage.get("prompt_tokens", 0)
            total_completion_tokens += usage.get("completion_tokens", 0)
            print(f"[CodingAgent] step {i+1}/{len(steps)}: {dt:.2f}s "
                  f"(prompt_tokens~{usage.get('prompt_tokens', 0)}, "
                  f"completion~{usage.get('completion_tokens', 0)})", flush=True)

        total = time.time() - session_start
        slo_met = (slo_seconds is None) or (total <= float(slo_seconds))

        result = {
            "agent": "coding",
            "num_steps": len(steps),
            "total_latency_s": total,
            "step_latencies_s": step_latencies,
            "slo_seconds": slo_seconds,
            "slo_met": slo_met,
            "total_prompt_tokens": total_prompt_tokens,
            "total_completion_tokens": total_completion_tokens,
        }

        if results_dir:
            try:
                os.makedirs(results_dir, exist_ok=True)
                with open(os.path.join(results_dir, f"coding_agent_{api_port}.json"), "w") as f:
                    json.dump(result, f, indent=2)
            except Exception as e:
                print(f"[CodingAgent] could not write result log: {e}", flush=True)

        print(f"[CodingAgent] DONE total={total:.2f}s slo={slo_seconds} met={slo_met}",
              flush=True)
        return result
