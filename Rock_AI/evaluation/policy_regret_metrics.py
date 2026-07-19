def utility_regret(best_utility: float, selected_utility: float) -> float:
    return max(0.0, float(best_utility) - float(selected_utility))
