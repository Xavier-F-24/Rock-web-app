import neat

from Rock_AI.training.neat_resume_helper import restore_population
from Rock_AI.training.recurrent_neat_training_helper import _write_config
from Rock_AI.neat.neat_genome_helper import BoundedRecurrentGenome


def test_checkpoint_restores_population_species_generation_and_innovations(tmp_path):
    config_path = tmp_path / "neat.ini"; _write_config(config_path, 4, 5)
    config = neat.Config(BoundedRecurrentGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, str(config_path))
    population = neat.Population(config, seed=51)
    original_keys = set(population.population)
    original_species = len(population.species.species)
    tracker = population.reproduction.innovation_tracker.global_counter
    prefix = str(tmp_path / "checkpoint-")
    neat.Checkpointer(1, filename_prefix=prefix).save_checkpoint(config, population.population, population.species, 7)
    restored = restore_population(f"{prefix}7")
    assert restored.generation == 7
    assert set(restored.population) == original_keys
    assert len(restored.species.species) == original_species
    assert restored.reproduction.innovation_tracker.global_counter == tracker
