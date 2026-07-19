"""Serialization-safe summaries of public transaction events."""

from dataclasses import asdict


def public_event_to_dict(event):
    return asdict(event)
