"""Canonical identifiers for replay-safe action candidates."""

import hashlib
import json
from typing import Iterable

from .farmer_action import FarmerAction


def canonical_action_hash(
    action: FarmerAction,
    *,
    observation_schema_version: int,
    action_schema_version: int,
    normalizer_version: int,
    public_rule_version: str,
    objective_values: Iterable[float],
    encoded_values: Iterable[float] = (),
    masks: Iterable[bool] = (),
) -> str:
    action_payload = action.to_dict()
    if action_payload.get("action_type") == "breed_pair":
        parent_ids = sorted((int(action_payload["parent_a_id"]), int(action_payload["parent_b_id"])))
        action_payload["parent_a_id"], action_payload["parent_b_id"] = parent_ids
        action_payload["potion_keys"] = sorted(action_payload.get("potion_keys", ()))
    payload = {
        "action": action_payload,
        "observation_schema_version": int(observation_schema_version),
        "action_schema_version": int(action_schema_version),
        "normalizer_version": int(normalizer_version),
        "public_rule_version": str(public_rule_version),
        "objective": list(map(float, objective_values)),
        "encoded_values": list(map(float, encoded_values)),
        "masks": list(map(bool, masks)),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()
