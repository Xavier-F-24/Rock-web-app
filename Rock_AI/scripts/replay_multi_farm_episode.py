import argparse
import json

from Rock_Serialization.rock_serialization_helper import world_from_dict


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode", required=True)
    args = parser.parse_args(argv)
    data = json.load(open(args.episode, encoding="utf-8"))
    initial = world_from_dict(data["initial_world"])
    print(f"Episode {data['episode_id']} seed={data['seed']} farms={len(initial.farms)}")
    for row in data["rounds"]:
        world = world_from_dict(row["world_after"])
        print(f"turn={row['world_turn']} generation={world.generation} order={','.join(row['acting_order'])}")
    final = world_from_dict(data["final_world"])
    final.validate_ownership()
    print(f"Replay snapshots valid. Final turn={final.turn}, rocks={len(final.owner_by_rock_id)}")


if __name__ == "__main__":
    main()
