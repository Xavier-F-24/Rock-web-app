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
    maximum_decisions_per_farm: int = 100
    maximum_no_progress_rounds: int = 8
    maximum_consecutive_passes: int = 6
    maximum_failed_transactions: int = 12
    maximum_episode_wall_clock_seconds: float = 300.0
    cycle_history_size: int = 12
    cycle_repeat_limit: int = 3
    heartbeat_interval_seconds: float = 5.0
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
    curriculum_max: ActionCurriculumStage = ActionCurriculumStage.OPPONENT_GENERALIZATION
    minimum_generations_per_stage: int = 3
    curriculum_stability_window: int = 3
    curriculum_invalid_rate_threshold: float = .05
    curriculum_validation_threshold: float = -.25

    def __post_init__(self):
        if min(self.population, self.generations, self.worlds_per_genome, self.max_rounds_per_world) <= 0:
            raise ValueError("Training counts must be positive")
        if min(
            self.maximum_decisions_per_farm,
            self.maximum_no_progress_rounds,
            self.maximum_consecutive_passes,
            self.maximum_failed_transactions,
            self.cycle_history_size,
            self.cycle_repeat_limit,
        ) <= 0:
            raise ValueError("Full-farmer liveness limits must be positive")
        if self.maximum_episode_wall_clock_seconds <= 0 or self.heartbeat_interval_seconds <= 0:
            raise ValueError("Timing limits must be positive")
        if self.worker_count != 1 and self.single_process:
            raise ValueError("single_process requires one worker")
        if self.curriculum_max < self.curriculum_start:
            raise ValueError("curriculum_max cannot precede curriculum_start")
        if min(self.minimum_generations_per_stage, self.curriculum_stability_window) <= 0:
            raise ValueError("Curriculum counts must be positive")

    def to_dict(self):
        data = asdict(self)
        data["curriculum_start"] = self.curriculum_start.name.lower()
        data["curriculum_max"] = self.curriculum_max.name.lower()
        return data

    @classmethod
    def from_dict(cls, data):
        payload = dict(data)
        payload["curriculum_start"] = ActionCurriculumStage[str(payload.get("curriculum_start", "imports")).upper()]
        payload["curriculum_max"] = ActionCurriculumStage[str(payload.get("curriculum_max", "opponent_generalization")).upper()]
        return cls(**payload)
