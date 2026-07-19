"""Stable superset feature schema for heterogeneous farmer actions."""

from dataclasses import dataclass

from .farmer_action_type import ACTION_TYPE_ORDER


ACTION_SCHEMA_VERSION = 1

SHARED_FEATURES = (
    "actor.money", "actor.committed_money", "actor.generation", "actor.rock_count",
    "actor.active_rock_count", "actor.potion_count", "world.turn", "world.listing_count",
    "world.offer_count", "capacity.remaining_breeding", "action.cost", "action.proceeds",
    "action.post_liquid_cash", "action.affordable", "action.reversible", "action.expiration_horizon",
    "result.success", "result.money_delta", "result.asset_delta", "result.delayed_resolution",
)
ROCK_FEATURES = (
    "rock_a.present", "rock_a.value", "rock_a.sell_value", "rock_a.generation",
    "rock_a.sex_male", "rock_a.sex_female", "rock_a.is_market", "rock_a.parent_count",
    "rock_b.present", "rock_b.value", "rock_b.sell_value", "rock_b.generation",
    "rock_b.sex_male", "rock_b.sex_female", "rock_b.is_market", "rock_b.parent_count",
    "pair.value_sum", "pair.value_difference", "pair.generation_difference", "pair.relatedness_r",
)
POTION_FEATURES = tuple(f"potion.type.{name}" for name in ("anti_craisen", "mutation", "fertility", "reroll")) + (
    "potion.price", "potion.inventory_count", "potion.mutation_effect", "potion.craisen_effect",
    "potion.clutch_effect", "potion.immediate_use",
)
MARKET_FEATURES = (
    "market.listing_price", "market.appraised_value", "market.price_to_value",
    "market.listing_age", "market.bid_count", "market.highest_visible_bid",
    "market.seller_public_value", "market.buyer_liquidity_after",
)
TRADE_FEATURES = (
    "trade.offered_rock_count", "trade.requested_rock_count", "trade.offered_money",
    "trade.requested_money", "trade.public_value_delta", "trade.liquidity_impact",
    "trade.offer_age", "trade.same_farm",
)
OBJECTIVE_FEATURES = (
    "objective.profit", "objective.diversity", "objective.rare_traits",
    "objective.mutation", "objective.max_value", "objective.risk_aversion",
)


@dataclass(frozen=True)
class ActionObservationSchema:
    version: int = ACTION_SCHEMA_VERSION

    @property
    def action_type_feature_names(self) -> tuple[str, ...]:
        return tuple(f"action_type.{action.value}" for action in ACTION_TYPE_ORDER)

    @property
    def feature_names(self) -> tuple[str, ...]:
        return self.action_type_feature_names + SHARED_FEATURES + ROCK_FEATURES + POTION_FEATURES + MARKET_FEATURES + TRADE_FEATURES + OBJECTIVE_FEATURES

    @property
    def feature_count(self) -> int:
        return len(self.feature_names)
