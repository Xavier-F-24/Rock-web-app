from dataclasses import asdict


def economy_timeline_rows(world):
    return tuple({"turn": event.world_turn, "type": event.event_type, "summary": event.summary, "farms": ", ".join(event.farm_ids), "rocks": ", ".join(map(str, event.rock_ids)), "payload": asdict(event)["payload"]} for event in world.public_events)
