"""Streamlit controller/view for the authoritative headless economy environment."""

from pathlib import Path

import streamlit as st

from Rock_AI.agents.full_neat_farmer_agent import FullNeatFarmerAgent
from Rock_AI.agents.heuristic_full_farmer_agent import HeuristicFullFarmerAgent
from Rock_AI.agents.random_full_farmer_agent import RandomFullFarmerAgent
from Rock_AI.environments.multi_farm_economy_environment import MultiFarmEconomyEnvironment
from Rock_AI.policies.recurrent_neat_farmer_policy import RecurrentNeatFarmerPolicy
from .action_candidate_panel import render_action_candidates
from .economy_training_panel import discover_full_farmer_runs, render_economy_training_panel
from .farmer_economy_panel import render_farmer_inventory
from .farmer_memory_panel import render_farmer_memory
from .multi_farm_comparison_panel import render_multi_farm_comparison
from .public_market_panel import render_public_market
from .trade_network_panel import render_trade_network
from .transaction_timeline_panel import render_transaction_timeline


ROOT = Path(__file__).resolve().parents[2]
ENV_KEY = "ai_obs_economy_environment"
AGENTS_KEY = "ai_obs_economy_agents"
ACTIVE_KEY = "ai_obs_economy_active_farm"


def _new_world(seed, champion):
    environment = MultiFarmEconomyEnvironment(int(seed))
    environment.reset()
    farm_ids = sorted(environment.world.farms)
    agents = {farm_ids[1]: HeuristicFullFarmerAgent(), farm_ids[2]: RandomFullFarmerAgent()}
    if champion:
        agents[farm_ids[0]] = FullNeatFarmerAgent(RecurrentNeatFarmerPolicy.load(champion))
    else:
        agents[farm_ids[0]] = HeuristicFullFarmerAgent("heuristic_lead")
    for index, (farm_id, agent) in enumerate(sorted(agents.items())):
        agent.reset(int(seed) + 10_000 + index, f"observatory-{seed}")
    st.session_state[ENV_KEY] = environment
    st.session_state[AGENTS_KEY] = agents
    st.session_state[ACTIVE_KEY] = farm_ids[0]


def _step_round(environment, agents):
    event_start = len(environment.world.public_events)
    selected = {}
    for farm_id, agent in agents.items():
        state = getattr(getattr(agent, "policy", None), "state", None)
        selected[farm_id] = agent.choose_candidate(environment.observe(farm_id, state))
    result = environment.resolve_round(selected)
    by_farm = {row.actor_farm_id: row for row in result.action_results}
    for farm_id, agent in agents.items():
        agent.observe_result(selected[farm_id], by_farm[farm_id])
    for event in environment.world.public_events[event_start:]:
        for agent in agents.values():
            if hasattr(agent, "observe_public_resolution"):
                agent.observe_public_resolution(event)
    return result


def render_world_economy_observatory():
    st.subheader("World Economy")
    runs = discover_full_farmer_runs()
    options = [None] + [run / "champions" / "best_validation" / "network.json" for run in runs]
    champion = st.selectbox("Lead farmer", options, format_func=lambda value: "Heuristic baseline" if value is None else value.parent.parent.parent.name, key="economy_champion")
    seed = st.number_input("World seed", min_value=0, value=2468, key="economy_world_seed")
    controls = st.columns(4)
    if controls[0].button("New World", key="economy_new_world"):
        _new_world(seed, champion)
        st.rerun()
    environment = st.session_state.get(ENV_KEY)
    agents = st.session_state.get(AGENTS_KEY)
    if environment is None:
        st.info("Create a world to watch three player-like farms share one market.")
        render_economy_training_panel()
        return
    if controls[1].button("Step World Round", disabled=environment.terminated, key="economy_step_round"):
        _step_round(environment, agents)
        st.rerun()
    if controls[2].button("Run Generation", disabled=environment.terminated, key="economy_run_generation"):
        start = environment.world.generation
        while not environment.terminated and environment.world.generation == start:
            _step_round(environment, agents)
        st.rerun()
    controls[3].metric("Turn / Generation", f"{environment.world.turn} / {environment.world.generation}")
    active = st.selectbox("Inspect farmer", sorted(environment.world.farms), format_func=lambda farm_id: environment.world.farm(farm_id).profile.display_name, key=ACTIVE_KEY)
    render_multi_farm_comparison(environment.world, active)
    farm_tab, market_tab, decision_tab, timeline_tab, trade_tab, memory_tab, training_tab = st.tabs(("Farm", "Public Market", "Action Decision", "Economy Timeline", "Trade Network", "Memory", "Training"))
    with farm_tab: render_farmer_inventory(environment.world, active)
    with market_tab: render_public_market(environment.world)
    with decision_tab: render_action_candidates(getattr(agents[active], "latest_decision", None))
    with timeline_tab: render_transaction_timeline(environment.world)
    with trade_tab: render_trade_network(environment.world)
    with memory_tab: render_farmer_memory(agents[active])
    with training_tab: render_economy_training_panel()
