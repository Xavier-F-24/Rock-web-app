"""Optional deterministic category cap, leaving final selection to one scorer."""

from collections import defaultdict


def cap_candidates_by_action_type(candidates, per_type_limit: int):
    grouped = defaultdict(list)
    for candidate in candidates:
        grouped[candidate.action.action_type].append(candidate)
    retained = []
    for action_type in sorted(grouped, key=lambda value: value.value):
        retained.extend(sorted(grouped[action_type], key=lambda row: row.candidate_hash)[:per_type_limit])
    return tuple(retained)
