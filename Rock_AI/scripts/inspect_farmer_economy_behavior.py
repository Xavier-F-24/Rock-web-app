import argparse
import json
from collections import Counter


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("episode")
    args = parser.parse_args(argv)
    data = json.load(open(args.episode, encoding="utf-8"))
    by_farm = {}
    for row in data["decisions"]:
        by_farm.setdefault(row["farm_id"], Counter())[row["selected_action"]["action_type"]] += 1
    for farm_id, counts in sorted(by_farm.items()):
        print(f"{farm_id}: {dict(counts)}")


if __name__ == "__main__":
    main()
