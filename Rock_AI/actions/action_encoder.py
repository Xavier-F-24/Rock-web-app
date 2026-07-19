"""Encode authoritative legal actions using player-visible data only."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Mapping

from Rock_Market.rock_market_helper import POTION_SHOP

from .action_candidate import ActionCandidate
from .action_hash import canonical_action_hash
from .action_schema import ActionObservationSchema
from .farmer_action import (
    AcceptBidAction, AcceptTradeOfferAction, BreedPairAction, BuyPotionAction,
    CancelListingAction, CreateListingAction, CreateTradeOfferAction, FarmerAction,
    ImportRandomRockAction, ImportRequestedRockAction, PlaceBidAction,
    RejectBidAction, RejectTradeOfferAction, SellRockAction,
)
from .farmer_action_type import ACTION_TYPE_ORDER


def _rock_row(rock: object | None) -> tuple[float, ...]:
    if rock is None:
        return (0.0,) * 8
    sex = str(getattr(getattr(rock, "sex", None), "value", getattr(rock, "sex", ""))).lower()
    return (
        1.0, float(getattr(rock, "value", 0)), float(getattr(rock, "sell_value", 0)),
        float(getattr(rock, "generation", 0)), float(sex == "male"), float(sex == "female"),
        float(bool(getattr(rock, "is_market", False))), float(len(getattr(rock, "parent_ids", ()) or ())),
    )


class ActionEncoder:
    def __init__(self, schema: ActionObservationSchema | None = None):
        self.schema = schema or ActionObservationSchema()

    def encode(
        self,
        action: FarmerAction,
        *,
        actor: object,
        world: object,
        objective: Mapping[str, float] | object,
        rock_a: object | None = None,
        rock_b: object | None = None,
        listing: object | None = None,
        relatedness_r: float = 0.0,
        reasons: tuple[str, ...] = (),
    ) -> ActionCandidate:
        objective_data = objective if isinstance(objective, Mapping) else asdict(objective)
        money = int(getattr(actor, "money", getattr(getattr(actor, "game", None), "money", 0)))
        committed = int(getattr(actor, "committed_money", 0))
        rocks = getattr(actor, "rocks", {})
        rock_values = rocks.values() if isinstance(rocks, dict) else rocks
        active = [r for r in rock_values if str(getattr(getattr(r, "status", None), "value", "")) == "active"]
        potions = getattr(actor, "potions", getattr(getattr(actor, "game", None), "potions", {}))
        action_cost = float(getattr(action, "quoted_cost", getattr(action, "bid_amount", getattr(action, "offered_money", 0))))
        proceeds = float(getattr(action, "quoted_sale_value", getattr(action, "requested_money", 0)))
        action_types = tuple(float(action.action_type == candidate) for candidate in ACTION_TYPE_ORDER)
        shared = (
            float(money), float(committed), float(getattr(actor, "generation", getattr(getattr(actor, "game", None), "generation", 0))),
            float(len(rocks)), float(len(active)), float(sum(potions.values())), float(getattr(world, "turn", 0)),
            float(len(getattr(world, "listings", {}))), float(len(getattr(world, "trade_offers", {}))),
            float(max(0, getattr(getattr(actor, "game", None), "max_pairs_per_generation", 0) - len(getattr(getattr(actor, "game", None), "breeding_queue", ())))),
            action_cost, proceeds, float(money - committed - action_cost + proceeds),
            float(money - committed >= action_cost), float(isinstance(action, (CreateListingAction, PlaceBidAction))),
            float(max(0, getattr(action, "expires_turn", getattr(listing, "expires_turn", 0)) - getattr(world, "turn", 0))),
            0.0, 0.0, 0.0, 0.0,
        )
        if isinstance(action, BreedPairAction) and rock_a is not None and rock_b is not None and int(rock_a.id) > int(rock_b.id):
            rock_a, rock_b = rock_b, rock_a
        left = _rock_row(rock_a)
        right = _rock_row(rock_b)
        rock_features = left + right + (
            left[1] + right[1], abs(left[1] - right[1]), abs(left[3] - right[3]), float(relatedness_r),
        )
        potion_key = getattr(action, "potion_type", None)
        potion = POTION_SHOP.get(potion_key, {})
        potion_features = tuple(float(potion_key == key) for key in ("anti_craisen", "mutation", "fertility", "reroll")) + (
            float(potion.get("cost", 0)), float(potions.get(potion_key, 0) if potion_key else 0),
            float(potion_key == "mutation"), float(potion_key == "anti_craisen"),
            float(potion_key in {"fertility", "reroll"}), float(isinstance(action, BreedPairAction) and bool(action.potion_keys)),
        )
        asking = float(getattr(listing, "asking_price", getattr(action, "asking_price", getattr(action, "bid_amount", 0))))
        appraisal = float(getattr(listing, "appraised_value", getattr(rock_a, "value", 0)))
        bids = getattr(listing, "bids", {}) if listing else {}
        market_features = (
            asking, appraisal, asking / max(1.0, appraisal),
            float(max(0, getattr(world, "turn", 0) - getattr(listing, "created_turn", getattr(world, "turn", 0)))),
            float(len(bids)), float(max((getattr(b, "amount", 0) for b in bids.values()), default=0)),
            float(getattr(listing, "seller_public_value", 0)), float(money - committed - action_cost),
        )
        offered_ids = getattr(action, "offered_rock_ids", ())
        requested_ids = getattr(action, "requested_rock_ids", ())
        trade_features = (
            float(len(offered_ids)), float(len(requested_ids)), float(getattr(action, "offered_money", 0)),
            float(getattr(action, "requested_money", 0)), float(getattr(action, "requested_money", 0) - getattr(action, "offered_money", 0)),
            float(getattr(action, "requested_money", 0) - getattr(action, "offered_money", 0)), 0.0,
            float(getattr(action, "recipient_farm_id", None) == action.actor_farm_id),
        )
        objective_features = tuple(float(objective_data.get(key, 0.0)) for key in (
            "profit_weight", "diversity_weight", "rare_trait_weight", "mutation_weight", "maximum_value_weight", "risk_aversion_weight",
        ))
        values = action_types + shared + rock_features + potion_features + market_features + trade_features + objective_features
        masks = tuple(True for _ in values)
        action_hash = canonical_action_hash(
            action, observation_schema_version=1, action_schema_version=self.schema.version,
            normalizer_version=1, public_rule_version=str(getattr(world, "rule_version", "1")),
            objective_values=objective_features, encoded_values=values, masks=masks,
        )
        return ActionCandidate(action, action_hash, values, masks, self.schema.feature_names, reasons)
