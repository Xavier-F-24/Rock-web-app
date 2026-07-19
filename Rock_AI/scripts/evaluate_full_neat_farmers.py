import argparse
import json
from pathlib import Path

from Rock_AI.evaluation.full_farmer_evaluator import FullFarmerEvaluator


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--champion", required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=5678)
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    result = FullFarmerEvaluator().evaluate(args.champion, episodes=args.episodes, seed=args.seed)
    print("Agent                         Mean utility    Std")
    for name, row in result["aggregate"].items():
        print(f"{name:28} {row['mean_utility']:12.3f} {row['std_utility']:8.3f}")
    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
