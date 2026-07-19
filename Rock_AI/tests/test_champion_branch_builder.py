import hashlib

import neat

from Rock_AI.neat.neat_export_helper import save_recurrent_artifact
from Rock_AI.neat.neat_genome_helper import BoundedRecurrentGenome
from Rock_AI.neat.neat_recurrent_network import RecurrentEvaluationConfig
from Rock_AI.neat.neat_topology_helper import (
    RECURRENT_TOPOLOGY_VERSION, RecurrentConnectionGene, RecurrentNodeGene,
    RecurrentTopologyArtifact, TopologyResourceLimits,
)
from Rock_AI.training.recurrent_neat_training_helper import _write_config
from Rock_AI.training_jobs.champion_branch_builder import build_champion_branch_population
from Rock_AI.training_jobs.training_job_config import TrainingJobConfig, TrainingOperation


def _artifact(path):
    artifact = RecurrentTopologyArtifact(
        RECURRENT_TOPOLOGY_VERSION, "parent-topology", 9, 2, 1, "player",
        (-1,-2,-3,-4), (0,1,2), ("a","b","c","d"),
        ("pair_score","stop_preference","confidence"),
        tuple(RecurrentNodeGene(index,"output",0.0,1.0,"tanh","sum") for index in range(3)),
        (RecurrentConnectionGene(-1,0,1.0,True,innovation_id=1),),
        RecurrentEvaluationConfig().to_dict(), TopologyResourceLimits().to_dict(), {},
    )
    save_recurrent_artifact(artifact,path)


def test_champion_branch_preserves_elite_and_adds_diversity(tmp_path):
    champion=tmp_path/"network.json"; _artifact(champion)
    before=hashlib.sha256(champion.read_bytes()).hexdigest()
    config_path=tmp_path/"neat.ini"; _write_config(config_path,4,10)
    neat_config=neat.Config(BoundedRecurrentGenome,neat.DefaultReproduction,neat.DefaultSpeciesSet,neat.DefaultStagnation,str(config_path))
    job=TrainingJobConfig(
        operation=TrainingOperation.BRANCH_CHAMPION,source_run=str(tmp_path),output_run=str(tmp_path/"branch"),
        additional_generations=1,seed=55,source_generation=4,source_champion=str(champion),population_size=10,
        training_scenarios=2,validation_scenarios=2,structural_mutation_scale=3.0,
    )
    population,manifest=build_champion_branch_population(job,neat_config,expected_schema=2,expected_features=("a","b","c","d"))
    assert len(population.population)==10
    elite=population.population[1]
    assert elite.connections[(-1,0)].weight==1.0
    signatures={tuple(sorted((key,gene.weight,gene.enabled) for key,gene in genome.connections.items())) for genome in list(population.population.values())[1:7]}
    assert len(signatures)>1
    assert manifest.exact_elite_count==1 and manifest.champion_descendant_count==6
    assert manifest.fresh_genome_count>0
    assert hashlib.sha256(champion.read_bytes()).hexdigest()==before
