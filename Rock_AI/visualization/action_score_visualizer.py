def action_score_rows(decision, limit=20):
    if decision is None:
        return ()
    return tuple({"rank": row.rank, "action": row.candidate.action.action_type.value, "target": row.candidate.action.to_dict(), "score": row.score, "selected": decision.selected is not None and row.candidate.candidate_hash == decision.selected.candidate_hash} for row in decision.ranked_actions[:limit])
