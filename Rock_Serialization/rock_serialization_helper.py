"""
Serialization helpers for the split-module GameMaster prototype.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import Rock_Genetics.rock_genetic_helper as genetics
from Rock_GameState.rock_game_state_helper import GameMaster, Inventory, QueuedPair
from Rock_GameState.rock_game_state_helper import DEFAULT_ROCK_FARM_COST
from Rock_Market.rock_market_helper import MarketPodOffer, PendingMarketPod


SAVE_VERSION = "0.3.0"


def allele_to_dict(allele: genetics.Allele) -> dict[str, int]:
    return {"value": int(allele.value)}


def allele_from_dict(data: dict[str, Any]) -> genetics.Allele:
    return genetics.Allele(value=int(data["value"]))


def gene_pair_to_dict(gene_pair: genetics.GenePair) -> dict[str, Any]:
    return {
        "allele_a": allele_to_dict(gene_pair.allele_a),
        "allele_b": allele_to_dict(gene_pair.allele_b),
        "name_of_gene": gene_pair.name_of_gene,
        "dominance_type": gene_pair.dominance_type,
        "money_value": int(gene_pair.money_value),
        "phenotype": gene_pair.phenotype,
    }


def gene_pair_from_dict(data: dict[str, Any]) -> genetics.GenePair:
    return genetics.GenePair(
        allele_a=allele_from_dict(data["allele_a"]),
        allele_b=allele_from_dict(data["allele_b"]),
        name_of_gene=str(data["name_of_gene"]),
        dominance_type=str(data["dominance_type"]),
        money_value=int(data.get("money_value", 0)),
        phenotype=data.get("phenotype"),
    )


def genome_to_dict(genome: genetics.Genome) -> dict[str, Any]:
    return {
        "genes": {
            gene_name: gene_pair_to_dict(gene_pair)
            for gene_name, gene_pair in genome.genes.items()
        }
    }


def genome_from_dict(data: dict[str, Any]) -> genetics.Genome:
    return genetics.Genome(
        genes={
            gene_name: gene_pair_from_dict(gene_data)
            for gene_name, gene_data in data.get("genes", {}).items()
        }
    )


def rock_name_to_dict(name: genetics.RockName | None) -> dict[str, Any] | None:
    if name is None:
        return None
    return {
        "given": name.given,
        "family": name.family,
        "honorific": name.honorific,
        "epithet": name.epithet,
    }


def rock_name_from_dict(data: dict[str, Any] | str | None) -> genetics.RockName | None:
    if data is None:
        return None
    if isinstance(data, str):
        return genetics.RockName(given=data)
    return genetics.RockName(
        given=str(data.get("given", "Rock")),
        family=data.get("family"),
        honorific=data.get("honorific"),
        epithet=data.get("epithet"),
    )


def rock_to_dict(rock: genetics.Rock) -> dict[str, Any]:
    return {
        "id": int(rock.id),
        "sex": rock.sex.value,
        "name": rock_name_to_dict(rock.name),
        "genotype": genome_to_dict(rock.genotype),
        "death_genes": genome_to_dict(rock.death_genes),
        "parent_ids": list(rock.parent_ids),
        "generation": int(rock.generation),
        "status": rock.status.value,
        "has_split": bool(rock.has_split),
        "checked_craisen": bool(rock.checked_craisen),
        "death_reason": rock.death_reason,
        "value": int(rock.value),
        "sell_value": int(rock.sell_value),
        "score_value": int(rock.score_value),
        "is_market": bool(rock.is_market),
    }


def rock_from_dict(data: dict[str, Any]) -> genetics.Rock:
    return genetics.Rock(
        id=int(data["id"]),
        sex=genetics.Sex(data["sex"]),
        name=rock_name_from_dict(data.get("name")),
        genotype=genome_from_dict(data.get("genotype", {})),
        death_genes=genome_from_dict(data.get("death_genes", {})),
        parent_ids=[int(parent_id) for parent_id in data.get("parent_ids", [])],
        generation=int(data.get("generation", 0)),
        status=genetics.RockStatus(data.get("status", genetics.RockStatus.ACTIVE.value)),
        has_split=bool(data.get("has_split", False)),
        checked_craisen=bool(data.get("checked_craisen", False)),
        death_reason=data.get("death_reason"),
        value=int(data.get("value", 0)),
        sell_value=int(data.get("sell_value", 0)),
        score_value=int(data.get("score_value", 0)),
        is_market=bool(data.get("is_market", False)),
    )


def inventory_to_dict(inventory: Inventory) -> dict[str, Any]:
    return {
        "money": int(inventory.money),
        "potions": dict(inventory.potions),
        "specials": dict(inventory.specials),
    }


def inventory_from_dict(data: dict[str, Any]) -> Inventory:
    return Inventory(
        money=int(data.get("money", 0)),
        potions=dict(data.get("potions", {})),
        specials=dict(data.get("specials", {})),
    )


def queued_pair_to_dict(pair: QueuedPair) -> dict[str, Any]:
    return {
        "parent_a_id": int(pair.parent_a_id),
        "parent_b_id": int(pair.parent_b_id),
        "potion_keys": list(pair.potion_keys),
        "potion_key": pair.potion_key,
    }


def queued_pair_from_dict(data: dict[str, Any]) -> QueuedPair:
    potion_keys = data.get("potion_keys")
    if potion_keys is None:
        old_potion_key = data.get("potion_key")
        potion_keys = [] if old_potion_key is None else [old_potion_key]
    elif isinstance(potion_keys, str):
        potion_keys = [potion_keys]

    return QueuedPair(
        parent_a_id=int(data["parent_a_id"]),
        parent_b_id=int(data["parent_b_id"]),
        potion_keys=list(potion_keys),
    )


def market_pod_to_dict(offer: MarketPodOffer) -> dict[str, Any]:
    return {
        "offer_id": offer.offer_id,
        "tier": offer.tier,
        "name": offer.name,
        "tagline": offer.tagline,
        "price": int(offer.price),
        "parent_a": rock_to_dict(offer.parent_a),
        "parent_b": rock_to_dict(offer.parent_b),
        "used": bool(offer.used),
    }


def market_pod_from_dict(data: dict[str, Any]) -> MarketPodOffer:
    return MarketPodOffer(
        offer_id=str(data["offer_id"]),
        tier=str(data["tier"]),
        name=str(data["name"]),
        tagline=str(data.get("tagline", "")),
        price=int(data["price"]),
        parent_a=rock_from_dict(data["parent_a"]),
        parent_b=rock_from_dict(data["parent_b"]),
        used=bool(data.get("used", False)),
    )


def pending_market_pod_to_dict(pending: PendingMarketPod | None) -> dict[str, Any] | None:
    if pending is None:
        return None
    return {
        "offer": market_pod_to_dict(pending.offer),
        "parent_a_id": int(pending.parent_a_id),
        "parent_b_id": int(pending.parent_b_id),
        "children": [rock_to_dict(child) for child in pending.children],
    }


def pending_market_pod_from_dict(data: dict[str, Any] | None) -> PendingMarketPod | None:
    if data is None:
        return None
    return PendingMarketPod(
        offer=market_pod_from_dict(data["offer"]),
        parent_a_id=int(data["parent_a_id"]),
        parent_b_id=int(data["parent_b_id"]),
        children=[rock_from_dict(child) for child in data.get("children", [])],
    )


def game_to_dict(game: GameMaster) -> dict[str, Any]:
    game.evaluate_all_rocks()
    return {
        "save_version": SAVE_VERSION,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "game": {
            "starting_money": int(game.starting_money),
            "max_generation": int(game.max_generation),
            "max_pairs_per_generation": int(game.max_pairs_per_generation),
            "rock_farm_cost": int(game.rock_farm_cost),
            "seed": game.seed,
            "generation": int(game.generation),
            "next_rock_id": int(game.next_rock_id),
            "game_over": bool(game.game_over),
            "inventory": inventory_to_dict(game.inventory),
            "breeding_queue": [queued_pair_to_dict(pair) for pair in game.breeding_queue],
            "events": list(game.events),
            "rocks": [rock_to_dict(rock) for _, rock in sorted(game.rock_list.items())],
            "market_pods": [market_pod_to_dict(offer) for offer in game.market_pods],
            "pending_market_pod": pending_market_pod_to_dict(game.pending_market_pod),
        },
    }


def game_from_dict(save_data: dict[str, Any]) -> GameMaster:
    if "game" not in save_data:
        raise ValueError("Invalid save data: missing game.")

    data = save_data["game"]
    game = GameMaster(
        starting_money=int(data.get("starting_money", 0)),
        max_generation=int(data.get("max_generation", 7)),
        max_pairs_per_generation=int(data.get("max_pairs_per_generation", 3)),
        rock_farm_cost=int(data.get("rock_farm_cost", DEFAULT_ROCK_FARM_COST)),
        seed=data.get("seed"),
        auto_start=False,
    )

    game.generation = int(data.get("generation", 0))
    game.next_rock_id = int(data.get("next_rock_id", 1))
    game.game_over = bool(data.get("game_over", False))
    game.inventory = inventory_from_dict(data.get("inventory", {}))
    game.breeding_queue = [
        queued_pair_from_dict(pair_data)
        for pair_data in data.get("breeding_queue", [])
    ]
    game.events = list(data.get("events", []))
    game.rock_list = {
        int(rock_data["id"]): rock_from_dict(rock_data)
        for rock_data in data.get("rocks", [])
    }
    game.market_pods = [
        market_pod_from_dict(offer_data)
        for offer_data in data.get("market_pods", [])
    ]
    game.pending_market_pod = pending_market_pod_from_dict(data.get("pending_market_pod"))

    if game.rock_list:
        game.next_rock_id = max(game.next_rock_id, max(game.rock_list) + 1)

    game.evaluate_all_rocks()
    return game


def game_to_json_string(game: GameMaster, indent: int = 2) -> str:
    return json.dumps(game_to_dict(game), indent=indent)


def game_from_json_string(json_string: str) -> GameMaster:
    return game_from_dict(json.loads(json_string))


def save_game_json(game: GameMaster, filepath: str | Path) -> Path:
    path = Path(filepath)
    path.write_text(game_to_json_string(game), encoding="utf-8")
    return path


def load_game_json(filepath: str | Path) -> GameMaster:
    return game_from_json_string(Path(filepath).read_text(encoding="utf-8"))


def make_save_filename(game: GameMaster) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"rock_game_gen{game.generation}_money{game.money}_{timestamp}.json"


def world_to_dict(world) -> dict[str, Any]:
    """Serialize the additive multi-farm economy without live policy objects."""
    from dataclasses import asdict

    serialized_games = {}
    for farm_id, farm in sorted(world.farms.items()):
        serialized = game_to_dict(farm.game)
        serialized.pop("saved_at", None)
        serialized_games[farm_id] = serialized
    return {
        "world_save_version": int(world.save_version),
        "seed": int(world.seed),
        "turn": int(world.turn),
        "generation": int(world.generation),
        "rule_version": world.rule_version,
        "farms": {
            farm_id: {
                "profile": farm.profile.to_dict(),
                "game": serialized_games[farm_id],
                "visible_rock_ids": sorted(farm.visible_rock_ids),
                "committed_money": int(farm.committed_money),
                "observable_history": list(farm.observable_history),
                "private_messages": list(farm.private_messages),
            }
            for farm_id, farm in sorted(world.farms.items())
        },
        "owner_by_rock_id": {str(key): value for key, value in world.owner_by_rock_id.items()},
        "reserved_rock_ids": {str(key): value for key, value in world.reserved_rock_ids.items()},
        "completed_transaction_ids": sorted(world.completed_transaction_ids),
        "listings": {
            key: {
                **{name: value for name, value in asdict(listing).items() if name != "bids"},
                "status": listing.status.value,
                "bids": {bid_id: {**asdict(bid)} for bid_id, bid in listing.bids.items()},
            }
            for key, listing in world.listings.items()
        },
        "trade_offers": {
            key: {**asdict(offer), "status": offer.status.value}
            for key, offer in world.trade_offers.items()
        },
        "messages": [asdict(message) for message in world.messages],
        "public_events": [asdict(event) for event in world.public_events],
    }


def world_from_dict(data: dict[str, Any]):
    from Rock_AI.logging.public_world_event_record import PublicWorldEventRecord
    from Rock_Market.rock_npc_market_helper import FarmMessage, ListingStatus, MarketBid, MarketListing, OfferStatus, TradeOffer
    from Rock_World.rock_farm_profile_helper import FarmObjective, FarmProfile
    from Rock_World.rock_world_state_helper import FarmState, WorldState

    farms = {}
    for farm_id, row in data.get("farms", {}).items():
        profile_data = dict(row["profile"])
        profile_data["objective"] = FarmObjective(profile_data["objective"])
        farms[farm_id] = FarmState(
            farm_id, FarmProfile(**profile_data), game_from_dict(row["game"]),
            set(map(int, row.get("visible_rock_ids", ()))), int(row.get("committed_money", 0)),
            list(row.get("observable_history", ())), list(row.get("private_messages", ())),
        )
    world = WorldState(
        farms, {int(key): value for key, value in data.get("owner_by_rock_id", {}).items()},
        int(data["seed"]), turn=int(data.get("turn", 0)), generation=int(data.get("generation", 0)),
        rule_version=str(data.get("rule_version", "economy-1")),
        save_version=int(data.get("world_save_version", 1)),
    )
    world.reserved_rock_ids = {int(key): value for key, value in data.get("reserved_rock_ids", {}).items()}
    world.completed_transaction_ids = set(data.get("completed_transaction_ids", ()))
    for listing_id, row in data.get("listings", {}).items():
        listing = MarketListing(
            listing_id, row["seller_farm_id"], int(row["rock_id"]), int(row["asking_price"]),
            int(row["appraised_value"]), int(row["created_turn"]), int(row["expires_turn"]),
            ListingStatus(row["status"]),
        )
        listing.bids = {bid_id: MarketBid(**bid) for bid_id, bid in row.get("bids", {}).items()}
        world.listings[listing_id] = listing
        world.bids.update(listing.bids)
    for offer_id, row in data.get("trade_offers", {}).items():
        payload = dict(row)
        payload["offered_rock_ids"] = tuple(payload.get("offered_rock_ids", ()))
        payload["requested_rock_ids"] = tuple(payload.get("requested_rock_ids", ()))
        payload["status"] = OfferStatus(payload["status"])
        world.trade_offers[offer_id] = TradeOffer(**payload)
    world.messages = [FarmMessage(**row) for row in data.get("messages", ())]
    world.public_events = [PublicWorldEventRecord(**row) for row in data.get("public_events", ())]
    world.validate_ownership()
    return world
