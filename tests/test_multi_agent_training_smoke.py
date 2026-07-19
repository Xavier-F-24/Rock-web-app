from Rock_AI.training.action_curriculum import ActionCurriculumStage, availability_for_stage
from Rock_AI.training.full_farmer_neat_trainer import FullFarmerNeatTrainer
from Rock_AI.training.full_farmer_training_config import FullFarmerTrainingConfig


def test_curriculum_uses_stable_schema_and_smoke_population_completes(tmp_path):
    widths = []
    for stage in ActionCurriculumStage:
        availability = availability_for_stage(stage)
        widths.append(len(availability.enabled))
    assert widths == sorted(widths)
    output = tmp_path / "full_farmer_smoke"
    config = FullFarmerTrainingConfig(
        str(output), seed=101, population=4, generations=1,
        worlds_per_genome=1, max_rounds_per_world=1,
        curriculum_start=ActionCurriculumStage.IMPORTS, checkpoint_frequency=1,
    )
    artifact, metadata = FullFarmerNeatTrainer(config).train()
    assert metadata["run_type"] == "recurrent_neat_full_farmer"
    assert artifact.metadata["policy_kind"] == "full_farmer"
    assert (output / "champions" / "best_validation" / "network.json").exists()
