from __future__ import annotations

from Rock_AI.environments.breeding_campaign_environment import BreedingCampaignEnvironment
from Rock_AI.visualization.farm_render_adapter import build_farm_rock_views, safe_render_rock_image


def test_farm_views_flag_selected_children_mutations_and_high_values():
    environment = BreedingCampaignEnvironment(seed=404)
    environment.reset(404)
    rock_ids = tuple(environment.game.rocks)
    views = build_farm_rock_views(
        environment.game,
        selected_parent_ids=(rock_ids[0],),
        new_child_ids=(rock_ids[1],),
        mutation_rock_ids=(rock_ids[1],),
        include_images=False,
    )
    by_id = {view.rock_id: view for view in views}
    assert by_id[rock_ids[0]].selected_parent
    assert by_id[rock_ids[1]].newly_created
    assert by_id[rock_ids[1]].mutated
    assert any(view.high_value for view in views)
    assert all(view.image_uri is None for view in views)


def test_missing_optional_image_does_not_break_farm_record_generation():
    environment = BreedingCampaignEnvironment(seed=405)
    environment.reset(405)

    def broken_renderer(*args, **kwargs):
        raise RuntimeError("renderer unavailable")

    rock = next(iter(environment.game.rocks.values()))
    assert safe_render_rock_image(rock, broken_renderer) is None
    views = build_farm_rock_views(environment.game, image_renderer=broken_renderer)
    assert views
    assert all(view.image_uri is None for view in views)
