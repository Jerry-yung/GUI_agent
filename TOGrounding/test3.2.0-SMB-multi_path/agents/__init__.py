"""Agent registry for test3.2.0-SMB multi-path."""
from __future__ import annotations

from agents.m2_agent import M2Agent
from agents.TO_agent import TOAgent
from agents.TOa_agent import TOaAgent
from agents.AppAgent import AppAgentAgent

AGENT_REGISTRY: dict[str, type] = {
    "m2": M2Agent,
    "to": TOAgent,
    "toa": TOaAgent,
    "AppAgent": AppAgentAgent,
}


def get_agent(name: str):
    if name not in AGENT_REGISTRY:
        raise ValueError(f"Unknown agent: {name!r}. Available: {list(AGENT_REGISTRY)}")
    return AGENT_REGISTRY[name]()


def load_agent(name: str, **kwargs):
    if name not in AGENT_REGISTRY:
        raise ValueError(f"Unknown agent: {name!r}. Available: {list(AGENT_REGISTRY)}")
    cls = AGENT_REGISTRY[name]
    if name in ("m2", "to", "toa"):
        return cls(top_k=kwargs.get("top_k", 10))
    return cls()
