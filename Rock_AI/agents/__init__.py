"""Breeding-only agents for deterministic headless campaigns."""

from .breeding_agent_helper import (
    AgentAction,
    BreedPairAction,
    BreedingAgent,
    CampaignObservation,
    NoAction,
    StopGenerationAction,
)
__all__ = [
    "AgentAction",
    "BreedPairAction",
    "BreedingAgent",
    "CampaignObservation",
    "HeuristicBreedingAgent",
    "NeuralBreedingAgent",
    "NoAction",
    "OracleBreedingAgent",
    "RandomBreedingAgent",
    "StopGenerationAction",
]


def __getattr__(name: str):
    modules = {
        "HeuristicBreedingAgent": "heuristic_breeding_agent",
        "NeuralBreedingAgent": "neural_breeding_agent",
        "OracleBreedingAgent": "oracle_breeding_agent",
        "RandomBreedingAgent": "random_breeding_agent",
    }
    if name in modules:
        from importlib import import_module

        return getattr(import_module(f"Rock_AI.agents.{modules[name]}"), name)
    raise AttributeError(name)
from .neat_breeding_agent import NeatBreedingAgent

__all__ = ["NeatBreedingAgent"]
from .recurrent_neat_breeding_agent import RecurrentNeatBreedingAgent
