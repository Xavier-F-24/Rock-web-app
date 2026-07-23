import math
import time

from Rock_AI.actions.action_mask import ActionAvailability
from Rock_AI.actions.farmer_action import (
    CreateListingAction,
    CreateTradeOfferAction,
    PassTurnAction,
    PlaceBidAction,
    RejectTradeOfferAction,
)
from Rock_AI.economy.reservation_audit_helper import audit_transaction_reservations
from Rock_AI.environments.episode_liveness_helper import (
    EpisodeLivenessLimits,
    EpisodeTerminationReason,
    world_progress_signature,
)
from Rock_AI.environments.multi_farm_economy_environment import (
    MultiFarmEconomyConfig,
    MultiFarmEconomyEnvironment,
)
from Rock_AI.policies.market_action_policy_adapter import (
    ActionCandidateLimits,
    LegalFarmerActionGenerator,
)
from Rock_AI.training_jobs.worker_heartbeat import (
    BackgroundHeartbeat, HeartbeatHealth,
    HeartbeatPhase,
    TimedHeartbeat,
    classify_heartbeat,
)
from Rock_Market.rock_npc_market_helper import ListingStatus, OfferStatus


def _install_generator(environment, generator):
    environment.candidate_generator = generator
    environment.observation_adapter.candidate_generator = generator


def _pass_candidates(environment):
    return {
        farm_id: next(
            row for row in environment.legal_candidates(farm_id)
            if isinstance(row.action, PassTurnAction)
        )
        for farm_id in environment.world.farms
    }


def test_no_legal_actions_still_exposes_pass_and_all_farms_blocked_terminates():
    environment = MultiFarmEconomyEnvironment(seed=201)
    environment.reset()
    _install_generator(
        environment,
        LegalFarmerActionGenerator(availability=ActionAvailability(frozenset())),
    )

    candidates = {
        farm_id: environment.legal_candidates(farm_id)
        for farm_id in environment.world.farms
    }
    assert all(len(rows) == 1 and isinstance(rows[0].action, PassTurnAction) for rows in candidates.values())

    result = environment.resolve_round({farm_id: rows[0] for farm_id, rows in candidates.items()})
    assert result.generation_advanced
    assert environment.world.turn == 1
    assert environment.termination_reason is EpisodeTerminationReason.ALL_FARMS_BLOCKED
    assert math.isfinite(environment.terminal_fitness())


def test_repeated_invalid_actions_fall_back_to_pass_and_consume_failure_budget():
    limits = EpisodeLivenessLimits(maximum_failed_transactions=1)
    environment = MultiFarmEconomyEnvironment(
        seed=202,
        config=MultiFarmEconomyConfig(liveness_limits=limits),
    )
    environment.reset()

    result = environment.resolve_round({farm_id: object() for farm_id in environment.world.farms})

    assert all(row.success for row in result.action_results)
    assert environment.world.turn == 1
    assert environment.liveness.failed_transactions == len(environment.world.farms)
    assert environment.termination_reason is EpisodeTerminationReason.MAX_FAILED_TRANSACTIONS


def test_all_rocks_reserved_in_listings_keeps_candidate_generation_bounded():
    environment = MultiFarmEconomyEnvironment(seed=203)
    environment.reset()
    farm_id = sorted(environment.world.farms)[0]
    farm = environment.world.farm(farm_id)
    for rock in list(farm.rocks.values()):
        if rock.status.value != "active":
            continue
        prices = environment.transaction_manager.legal_price_menu(rock.value)
        action = CreateListingAction(farm_id, environment.world.turn, rock.id, prices[0])
        result = environment.transaction_manager.execute(
            environment.world,
            action,
            f"reserve-all-{farm_id}-{rock.id}",
        )
        assert result.success

    candidates = environment.legal_candidates(farm_id)
    assert candidates
    assert any(isinstance(row.action, PassTurnAction) for row in candidates)
    assert len(candidates) <= environment.config.candidate_limits.maximum_total_legal_actions


def test_expired_bid_releases_committed_cash_and_listing_reservation():
    environment = MultiFarmEconomyEnvironment(seed=204)
    environment.reset()
    seller_id, buyer_id = sorted(environment.world.farms)[:2]
    listing = next(row for row in environment.legal_candidates(seller_id) if isinstance(row.action, CreateListingAction))
    assert environment.execute(listing).success
    bid = next(row for row in environment.legal_candidates(buyer_id) if isinstance(row.action, PlaceBidAction))
    assert environment.execute(bid).success
    record = environment.world.listings[bid.action.listing_id]
    assert environment.world.farm(buyer_id).committed_money == bid.action.bid_amount

    record.expires_turn = environment.world.turn - 1
    report = audit_transaction_reservations(environment.world, repair=True)

    assert report.repaired
    assert record.status is ListingStatus.EXPIRED
    assert environment.world.farm(buyer_id).committed_money == 0
    assert record.rock_id not in environment.world.reserved_rock_ids
    assert all(not row.active for row in record.bids.values())


def test_rejected_trade_releases_offered_rocks_and_money():
    environment = MultiFarmEconomyEnvironment(seed=205)
    environment.reset()
    sender_id, recipient_id = sorted(environment.world.farms)[:2]
    offer = next(
        row for row in environment.legal_candidates(sender_id)
        if isinstance(row.action, CreateTradeOfferAction)
        and row.action.recipient_farm_id == recipient_id
    )
    assert environment.execute(offer).success
    offer_record = next(iter(environment.world.trade_offers.values()))
    offered = offer_record.offered_rock_ids[0]
    reject = next(
        row for row in environment.legal_candidates(recipient_id)
        if isinstance(row.action, RejectTradeOfferAction)
    )

    assert environment.execute(reject).success
    assert offer_record.status is OfferStatus.REJECTED
    assert offered not in environment.world.reserved_rock_ids
    assert environment.world.farm(sender_id).committed_money == 0


def test_state_cycle_and_episode_timeout_are_typed_finite_exits():
    now = [0.0]
    limits = EpisodeLivenessLimits(
        maximum_wall_clock_seconds=1.0,
        cycle_repeat_limit=3,
    )
    environment = MultiFarmEconomyEnvironment(
        seed=206,
        config=MultiFarmEconomyConfig(liveness_limits=limits),
        clock=lambda: now[0],
    )
    environment.reset()
    signature = world_progress_signature(environment.world)
    environment.liveness.record_signature(signature)
    environment.liveness.record_signature(signature)
    environment._check_termination()
    assert environment.termination_reason is EpisodeTerminationReason.STATE_CYCLE
    assert math.isfinite(environment.terminal_fitness(float("nan")))

    environment.reset()
    now[0] = 2.0
    environment._check_termination()
    assert environment.termination_reason is EpisodeTerminationReason.WALL_CLOCK_TIMEOUT


def test_candidate_generation_caps_each_expensive_action_family():
    limits = ActionCandidateLimits(
        maximum_active_rocks_considered=4,
        maximum_breeding_pairs=2,
        maximum_listing_actions=2,
        maximum_bid_actions=1,
        maximum_trade_actions=1,
        maximum_trade_bundles=1,
        maximum_total_legal_actions=12,
    )
    environment = MultiFarmEconomyEnvironment(
        seed=207,
        config=MultiFarmEconomyConfig(candidate_limits=limits),
    )
    environment.reset()
    farm_id = sorted(environment.world.farms)[0]
    rows = environment.legal_candidates(farm_id)
    counts = environment.candidate_generator.last_pruning_record["counts_by_type"]

    assert len(rows) <= 12
    assert counts["breed_pair"] <= 2
    assert counts["create_listing"] <= 2
    assert counts["create_trade_offer"] <= 1
    assert counts["pass_turn"] == 1


def test_time_based_heartbeat_fires_during_deliberately_slow_scenario():
    events = []
    heartbeat = TimedHeartbeat(events.append, interval_seconds=0.01)
    heartbeat.pulse(HeartbeatPhase.SCENARIO_EVALUATION, force=True, operation="started")
    time.sleep(0.02)
    heartbeat.pulse(HeartbeatPhase.SCENARIO_EVALUATION, operation="still_running")

    assert [row["operation"] for row in events] == ["started", "still_running"]
    assert all(row["health"] == HeartbeatHealth.HEALTHY.value for row in events)
    assert classify_heartbeat(40, process_alive=True) is HeartbeatHealth.SLOW
    assert classify_heartbeat(130, process_alive=True) is HeartbeatHealth.STAGNANT
    assert classify_heartbeat(1, process_alive=False) is HeartbeatHealth.ORPHANED
    assert classify_heartbeat(1, failed=True) is HeartbeatHealth.FAILED

    background_events = []
    background = BackgroundHeartbeat(background_events.append, interval_seconds=0.01)
    background.update(HeartbeatPhase.SCENARIO_EVALUATION, operation="deliberately_slow")
    background.start()
    time.sleep(0.035)
    background.stop()
    assert background_events
    assert all(row["phase"] == HeartbeatPhase.SCENARIO_EVALUATION.value for row in background_events)
