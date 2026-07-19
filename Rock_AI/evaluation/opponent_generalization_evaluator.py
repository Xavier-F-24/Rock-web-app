def summarize_objective_generalization(rows):
    objectives = sorted({row.get("objective", "unknown") for row in rows})
    return {name: sum(row.get("objective_utility", 0.0) for row in rows if row.get("objective", "unknown") == name) for name in objectives}
