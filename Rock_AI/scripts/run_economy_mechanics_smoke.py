"""Deterministically exercise every currently supported economy action family."""

import argparse
import json
from pathlib import Path

from Rock_AI.actions.farmer_action import (
    AcceptBidAction, AcceptTradeOfferAction, BreedPairAction, BuyPotionAction,
    CancelListingAction, CreateListingAction, CreateTradeOfferAction,
    ImportRandomRockAction, ImportRequestedRockAction, PassTurnAction, PlaceBidAction,
    RejectBidAction, RejectTradeOfferAction, SellRockAction, StopBreedingAction,
)
from Rock_AI.environments.multi_farm_economy_environment import MultiFarmEconomyEnvironment


def select(environment, farm_id, action_class, predicate=lambda action: True):
    return next(row for row in environment.legal_candidates(farm_id) if isinstance(row.action, action_class) and predicate(row.action))


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument("--output", default="evaluation_runs/economy_mechanics_smoke.json")
    args = parser.parse_args(argv)
    environment = MultiFarmEconomyEnvironment(args.seed)
    environment.reset()
    farm_a, farm_b, farm_c = sorted(environment.world.farms)
    results = []

    def execute(candidate):
        result = environment.execute(candidate)
        if not result.success:
            raise RuntimeError(result.summary)
        results.append({"action": candidate.action.to_dict(), "result": {"summary": result.summary, "payload": result.public_payload}})
        return result

    execute(select(environment, farm_a, BuyPotionAction, lambda action: action.potion_type == "mutation"))
    execute(select(environment, farm_a, ImportRandomRockAction))
    execute(select(environment, farm_a, ImportRequestedRockAction))
    execute(select(environment, farm_a, BreedPairAction, lambda action: action.potion_keys == ("mutation",)))
    execute(select(environment, farm_a, SellRockAction))

    listing = execute(select(environment, farm_b, CreateListingAction))
    listing_id = listing.public_payload["listing_id"]
    execute(select(environment, farm_c, PlaceBidAction, lambda action: action.listing_id == listing_id))
    execute(select(environment, farm_b, AcceptBidAction, lambda action: action.listing_id == listing_id))

    cancellable = execute(select(environment, farm_b, CreateListingAction))
    execute(select(environment, farm_b, CancelListingAction, lambda action: action.listing_id == cancellable.public_payload["listing_id"]))

    rejectable = execute(select(environment, farm_c, CreateListingAction))
    execute(select(environment, farm_b, PlaceBidAction, lambda action: action.listing_id == rejectable.public_payload["listing_id"]))
    execute(select(environment, farm_c, RejectBidAction, lambda action: action.listing_id == rejectable.public_payload["listing_id"]))

    trade = execute(select(environment, farm_a, CreateTradeOfferAction, lambda action: action.recipient_farm_id == farm_c))
    execute(select(environment, farm_c, AcceptTradeOfferAction, lambda action: action.offer_id == trade.public_payload["offer_id"]))
    rejected_trade = execute(select(environment, farm_b, CreateTradeOfferAction, lambda action: action.recipient_farm_id == farm_c))
    execute(select(environment, farm_c, RejectTradeOfferAction, lambda action: action.offer_id == rejected_trade.public_payload["offer_id"]))

    execute(select(environment, farm_a, StopBreedingAction))

    passes = {farm_id: select(environment, farm_id, PassTurnAction) for farm_id in environment.world.farms}
    round_result = environment.resolve_round(passes)
    results.extend({"action": passes[row.actor_farm_id].action.to_dict(), "result": {"summary": row.summary, "payload": row.public_payload}} for row in round_result.action_results)
    environment.world.validate_ownership()
    action_types = sorted({row["action"]["action_type"] for row in results})
    payload = {"seed": args.seed, "action_types": action_types, "results": results, "final_rock_count": len(environment.world.owner_by_rock_id)}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Executed {len(results)} actions across {len(action_types)} action types.")
    print(", ".join(action_types))
    print(f"Artifact: {output}")


if __name__ == "__main__":
    main()
