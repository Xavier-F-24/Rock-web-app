from Rock_Streamlit.components.champion_branch_panel import branch_population_counts
from Rock_Streamlit.components.completed_generation_panel import comparison_deltas, topology_diff
from Rock_Streamlit.components.neat_training_console import launch_enabled
from Rock_Streamlit.components.training_configuration_panel import estimated_evaluations, sanitize_run_name


class Capabilities:
    subprocess_supported=True; persistent_storage_supported=True; writable_training_directory=True


def test_review_helpers_are_deterministic():
    assert estimated_evaluations(10, 2, 5, 3) == 160
    assert sanitize_run_name(" My branch! ") == "My_branch"
    assert branch_population_counts(20, 1, .6, .15) == {"elite": 1, "descendants": 12, "historical": 3, "fresh": 4}


def test_launch_control_requires_confirmation_and_no_active_writer():
    assert launch_enabled(Capabilities(), True, False)
    assert not launch_enabled(Capabilities(), False, False)
    assert not launch_enabled(Capabilities(), True, True)


def test_topology_and_metric_comparison():
    parent={"nodes":[{"node_id":0,"activation":"tanh","aggregation":"sum"}],"connections":[]}
    child={"nodes":[{"node_id":0,"activation":"relu","aggregation":"sum"},{"node_id":1,"activation":"tanh","aggregation":"sum"}],"connections":[{"source_id":0,"target_id":1,"recurrent":True}]}
    diff=topology_diff(parent,child)
    assert diff["added_nodes"] == [1] and diff["changed_activations"] == [0]
    assert diff["recurrent_edges_added"] == [(0,1)]
    assert comparison_deltas({"best_fitness":.2},{"best_fitness":.5})["best_fitness"] == .3
