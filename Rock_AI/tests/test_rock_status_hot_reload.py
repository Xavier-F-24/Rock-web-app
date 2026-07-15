from __future__ import annotations

from enum import Enum

import pytest

import Rock_Genetics.rock_genetic_helper as genetics
from Rock_AI.environments.breeding_campaign_environment import BreedingCampaignEnvironment


class LegacyRockStatus(str, Enum):
    BRED = "bred"


def test_change_status_coerces_equivalent_hot_reloaded_enum():
    environment = BreedingCampaignEnvironment(seed=914)
    environment.reset(914)
    rock = next(iter(environment.game.rocks.values()))

    rock.change_status(LegacyRockStatus.BRED)

    assert rock.status is genetics.RockStatus.BRED
    assert not rock.is_active


def test_change_status_rejects_unknown_status_value():
    environment = BreedingCampaignEnvironment(seed=915)
    environment.reset(915)
    rock = next(iter(environment.game.rocks.values()))

    with pytest.raises(TypeError, match="RockStatus"):
        rock.change_status("not-a-status")
