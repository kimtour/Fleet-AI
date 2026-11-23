"""Tiny explicit state-graph runtime with the LangGraph API subset we use.

`build(nodes, edges, conditional, entry, backend)`:
  * backend="native"    - dependency-free runner below (default; what the evals exercise)
  * backend="langgraph" - same node functions compiled with langgraph.StateGraph
                          (requires `pip install langgraph`; see README for its test status)
Nodes are `state -> partial_update` functions; conditional edges are `state -> node_name`.
"""
from __future__ import annotations

from typing import Any, Callable, TypedDict

END = "__end__"


class AgentState(TypedDict, total=False):
    """Declared channels for the LangGraph backend. LangGraph drops keys a node returns that are not
    declared in the state schema, so every key any node reads or writes must be listed here
    (tests/test_suite.py asserts this against a real run)."""
    question: str
    obs: list
    cursor: int
    steps: list
    status: str
    expanded: bool
    claims: list
    notes: list
    plan: Any
    refusal: str
    answer: Any


class NativeGraph:
    def __init__(self, nodes: dict[str, Callable], edges: dict[str, str], conditional: dict[str, Callable],
                 entry: str, max_hops: int = 200):
        self.nodes, self.edges, self.cond, self.entry, self.max_hops = nodes, edges, conditional, entry, max_hops

    def invoke(self, state: dict) -> dict:
        state, node, hops = dict(state), self.entry, 0
        while node != END:
            hops += 1
            if hops > self.max_hops:                       # hard stop against runaway loops
                state["status"] = "aborted_step_budget"
                break
            state.update(self.nodes[node](state) or {})
            node = self.cond[node](state) if node in self.cond else self.edges[node]
        return state


def resolve_backend(backend: str) -> str:
    """'auto' = LangGraph when installed, else the dependency-free native runner (same node functions)."""
    if backend != "auto":
        return backend
    try:
        import langgraph  # type: ignore  # noqa: F401
        return "langgraph"
    except ImportError:
        return "native"


def build(nodes, edges, conditional, entry, backend: str = "native"):
    backend = resolve_backend(backend)
    if backend == "native":
        return NativeGraph(nodes, edges, conditional, entry)
    if backend == "langgraph":
        from langgraph.graph import END as LG_END, StateGraph  # type: ignore

        g = StateGraph(AgentState)
        for name, fn in nodes.items():
            g.add_node(name, fn)
        g.set_entry_point(entry)
        for src, dst in edges.items():
            g.add_edge(src, LG_END if dst == END else dst)
        for src, fn in conditional.items():
            g.add_conditional_edges(src, lambda s, fn=fn: (LG_END if fn(s) == END else fn(s)))
        return g.compile()
    raise ValueError(f"unknown backend {backend}")
