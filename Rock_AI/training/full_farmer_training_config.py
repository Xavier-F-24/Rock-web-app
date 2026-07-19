from dataclasses import asdict, dataclass

from .action_curriculum import ActionCurriculumStage


@dataclass(frozen=True)
class FullFarmerTrainingConfig:
    output_directory: str
    seed: int = 1234
    population: int = 100
    generations: int = 50
    worlds_per_genome: int = 3
    max_rounds_per_world: int = 6
    curriculum_start: ActionCurriculumStage = ActionCurriculumStage.IMPORTS
    checkpoint_frequency: int = 5
    showcase_frequency: int = 1
    settling_steps: int = 3
    worker_count: int = 1
    single_process: bool = True
    campaign_weight: float = .60
    relative_weight: float = .20
    market_weight: float = .10
    action_ranking_weight: float = .05
    memory_weight: float = .05
    complexity_penalty: float = .00001

    def __post_init__(self):
        if min(self.population, self.generations, self.worlds_per_genome, self.max_rounds_per_world) <= 0:
            raise ValueError("Training counts must be positive")
        if self.worker_count != 1 and self.single_process:
            raise ValueError("single_process requires one worker")

    def to_dict(self):
        data = asdict(self)
        data["curriculum_start"] = self.curriculum_start.name.lower()
        return data
