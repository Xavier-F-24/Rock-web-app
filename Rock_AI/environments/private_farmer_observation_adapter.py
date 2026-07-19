"""Construct one fresh player-visible observation per acting farm."""

from dataclasses import asdict

from Rock_AI.observations.full_farmer_observation import FullFarmerObservation
from Rock_AI.observations.player_economy_observation import ECONOMY_OBSERVATION_SCHEMA_VERSION, PlayerEconomyObservation
from Rock_AI.observations.player_inventory_observation import PlayerInventoryObservation
from Rock_AI.observations.player_market_observation import PlayerMarketObservation
from Rock_AI.observations.player_offer_observation import PlayerOfferObservation
from Rock_AI.observations.player_potion_observation import PlayerPotionObservation

from .public_information_adapter import PublicInformationAdapter


class PrivateFarmerObservationAdapter:
    """The only world-to-full-farmer policy boundary."""

    def __init__(self, candidate_generator):
        self.candidate_generator = candidate_generator
        self.public = PublicInformationAdapter()

    def build(self, world, farm_id: str, *, recurrent_state=None) -> FullFarmerObservation:
        farm = world.farm(farm_id)
        summaries = tuple(
            (int(rock.id), float(rock.value), float(rock.sell_value), int(rock.generation), str(rock.sex.value), str(rock.status.value))
            for rock in sorted(farm.rocks.values(), key=lambda rock: rock.id)
        )
        inventory = PlayerInventoryObservation(
            farm_id, farm.money, farm.committed_money, tuple(sorted(farm.rocks)), summaries,
            tuple(sorted((str(key), int(value)) for key, value in farm.potions.items())),
        )
        listings = self.public.listings(world)
        market = PlayerMarketObservation(
            listings, self.public.random_import_cost(), self.public.potion_shop(),
            sum(row.asking_price for row in listings) / len(listings) if listings else 0.0,
        )
        incoming = tuple(sorted(offer.offer_id for offer in world.trade_offers.values() if offer.recipient_farm_id == farm_id and offer.status.value == "open"))
        outgoing = tuple(sorted(offer.offer_id for offer in world.trade_offers.values() if offer.sender_farm_id == farm_id and offer.status.value == "open"))
        own_bids = tuple(sorted(bid.bid_id for bid in world.bids.values() if bid.bidder_farm_id == farm_id and bid.active))
        offers = PlayerOfferObservation(incoming, outgoing, own_bids, ())
        potions = PlayerPotionObservation(inventory.potions, market.potion_shop)
        opponents = self.public.public_farms(world, farm_id)
        payload = {
            "schema_version": ECONOMY_OBSERVATION_SCHEMA_VERSION, "actor_farm_id": farm_id,
            "world_turn": world.turn, "generation": farm.generation, "inventory": asdict(inventory),
            "market": asdict(market), "opponents": [asdict(row) for row in opponents],
            "potions": asdict(potions), "offers": asdict(offers), "public_rule_version": world.rule_version,
        }
        economy = PlayerEconomyObservation(
            ECONOMY_OBSERVATION_SCHEMA_VERSION, farm_id, world.turn, farm.generation,
            inventory, market, opponents, potions, offers, world.rule_version,
            self.public.public_hash(payload),
        )
        return FullFarmerObservation(economy, self.candidate_generator.generate(world, farm_id), recurrent_state)
