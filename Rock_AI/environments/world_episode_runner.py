"""Synchronous multi-agent episode orchestration with private observations."""

from __future__ import annotations

import copy
import json
from dataclasses import asdict

from Rock_AI.logging.market_decision_record import MarketDecisionRecord
from Rock_AI.logging.multi_farm_episode_record import MultiFarmEpisodeRecord
from Rock_Serialization.rock_serialization_helper import world_to_dict


class MultiFarmEpisodeRunner:
    def __init__(self, environment, agents: dict[str, object]):
        self.environment = environment
        self.agents = agents

    def run(self, *, seed: int, max_rounds: int | None = None):
        world = self.environment.reset(seed=seed)
        episode_id = f"economy-{seed}"
        for index, (farm_id, agent) in enumerate(sorted(self.agents.items())):
            agent.reset(seed + 10_000 + index, episode_id)
        record = MultiFarmEpisodeRecord(episode_id, seed, "mixed_economy", world_to_dict(world))
        limit = max_rounds or self.environment.config.max_world_turns
        while not self.environment.terminated and len(record.rounds) < limit:
            event_start = len(world.public_events)
            selected = {}
            observations = {}
            for farm_id in sorted(world.farms):
                agent = self.agents[farm_id]
                state = agent.policy.state if hasattr(agent, "policy") else None
                observation = self.environment.observe(farm_id, recurrent_state=copy.deepcopy(state))
                candidate = agent.choose_candidate(observation)
                if candidate is None:
                    raise RuntimeError(f"Agent {agent.name} returned no candidate despite authoritative pass action")
                observations[farm_id] = observation
                selected[farm_id] = candidate
            result = self.environment.resolve_round(selected)
            result_by_farm = {row.actor_farm_id: row for row in result.action_results}
            for farm_id in sorted(world.farms):
                agent = self.agents[farm_id]
                outcome = result_by_farm[farm_id]
                agent.observe_result(selected[farm_id], outcome)
                decision = getattr(agent, "latest_decision", None)
                ranked = () if decision is None else tuple({
                    "rank": row.rank, "action_hash": row.candidate.candidate_hash,
                    "action_type": row.candidate.action.action_type.value, "score": row.score,
                } for row in decision.ranked_actions[:10])
                record.decisions.append(MarketDecisionRecord(
                    episode_id, result.world_turn, farm_id, observations[farm_id].economy.observation_hash,
                    len(observations[farm_id].legal_candidates), selected[farm_id].candidate_hash,
                    selected[farm_id].action.to_dict(), ranked,
                    {"success": outcome.success, "summary": outcome.summary, "payload": outcome.public_payload},
                    decision.model_trace.get("memory_before") if decision and decision.model_trace else None,
                    agent.policy.export_state() if hasattr(agent, "policy") else None,
                ).to_dict())
            for event in world.public_events[event_start:]:
                for agent in self.agents.values():
                    if hasattr(agent, "observe_public_resolution"):
                        agent.observe_public_resolution(event)
            record.rounds.append({
                "world_turn": result.world_turn, "acting_order": list(result.acting_order),
                "generation_advanced": result.generation_advanced,
                "world_after": world_to_dict(world),
            })
        record.final_world = world_to_dict(world)
        record.termination_reason = self.environment.termination_reason or "round_limit"
        return record


def write_episode_jsonl(record, path):
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(json.dumps({"record_type": "manifest", **record.to_dict()}, sort_keys=True) + "\n")
