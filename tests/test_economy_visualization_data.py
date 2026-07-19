from Rock_AI.visualization.farmer_comparison_visualizer import farmer_comparison_rows
from Rock_AI.visualization.market_flow_visualizer import money_asset_flow_rows
from Rock_World import create_starter_world


def test_world_economy_visualization_uses_public_serializable_rows():
    world = create_starter_world(seed=121)
    farms = farmer_comparison_rows(world)
    flow = money_asset_flow_rows(world)
    assert len(farms) == len(flow) == 3
    assert all("genotype" not in key for row in farms for key in row)
    assert {row["farm_id"] for row in farms} == set(world.farms)
