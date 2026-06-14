#-----------------------------------------------------
"""
Rock Genetic Helper 

This file answers:

- What is a Rock?
- what genes does it have
    - how do we create the genes
    - random or parents?
        - parents if so
- what is the calculated phenotype (once there)
    - tricksy outside import??
- image of the rock once generated
- monetary value of the rock
- status of the rock in the world

- methods to:
    - change status
    - change image
    - change genotype
    - instantiate through different types
    - mitosion and spor handling
    - delete self -> unused market rocks
    - determine phenotype (possibly)

"""
#-----------------------------------------------------

#-----------------------------------------------------
# IMPORT ZONE
#-----------------------------------------------------

import math, random, base64, io

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Tuple, Optional, Any

#-----------------------------------------------------
# ROCK GENES ZONE
#-----------------------------------------------------

Rock_gene_dict = { # "gene": [[roll],
                            #[trait],
                            #[name],
                            #[cost],
                            #[dominance]]

    "shape": [[1,17,18,19,20],
              [0,1,2,3,4],
              ["circle","oval","square","triangle","oblong"],
              [0,-1,2,4,-2],
              [0,1,2,3,4]],

    "size": [[1,17,18,19,20],
             [0,1,2,3,4],
             ["medium","large","small","giant","missized"],
             [0,1,2,4,-2],
             [0,1,2,3,4]],

    "color": [[1,11,16,17,18,19,20],
              [0,1,2,3,4,5,6],
              ["white","black","brown","red","yellow","blue","patchwork"],
              [0,0,1,2,2,2,-1],
              [0,0,1,2,2,2,3]],

    "eyes": [[1,18],
             [0,1],
             ["n/a","eye","double eye"],
             [0,1,2],
             [0,0],],

    "brows": [[1,18,19,20],
              [0,1,2,3],
              ["n/a","brows","eyehair","unibrows"],
              [0,1,2,-1],
              [0,1,2,3]],

    "mouths": [[1,17,18,19,20],
               [0,1,2,3,4],
               ["n/a","mouth","smile","chip","smeagol"],
               [0,1,2,3,-1],
               [0,1,2,3,4]],

    "noses": [[1,17,18,19,20],
              [0,1,2,3,4],
              ["n/a","nub","honk","holes","concave"],
              [0,1,2,3,-1],
              [0,1,2,3,4]],

    "arms": [[1,19,20],
             [0,1,2],
             ["n/a","arms","muscle arms"],
             [0,1,2],
             [0,0,0]],

    "crowns": [[1,17,18,19,20],
               [0,1,2,3,4],
               ["n/a","small","medium","large","indent"],
               [0,1,2,3,-1],
               [0,1,2,3,4]],

    "wings": [[1,20],
              [0,1],
              ["n/a","wings"],
              [0,2],
              [0,1]],

    "halos": [[1,20],
              [0,1],
              ["n/a","halos"],
              [0,2],
              [0,1]],

    "horns": [[1,20],
              [0,1],
              ["n/a","horns"],
              [0,2],
              [0,1]],

    "wrinkles": [[1,20],
                 [0,1],
                 ["n/a","wrinkles"],
                 [0,2],
                 [0,1]],

    "fuzz": [[1,20],
             [0,1],
             ["n/a","fuzzy","spiky"],
             [0,1,2],
             [0,1]],

    "hair": [[1,18],
             [0,1],
             ["n/a","hair","double hair"],
             [0,1,2],
             [0,1]],

    "facial_hair": [[0,15,16,17,18,19,20],
                    [0,1,2,3,4,5,6],
                    ["n/a","goatee","beard","pedo","curly","chapman","sol"],
                    [0,1,2,-1,3,4,-2],
                    [0,1,2,3,4,5,6]],

    "freckles": [[1,20],
                 [0,1],
                 ["n/a","freckles"],
                 [0,2],
                 [0,1]],

    "stones": [[1,20],
               [0,1],
               ["n/a","stones"],
               [0,2],
               [0,1]],

    "tails": [[1,20],
              [0,1],
              ["n/a","tails"],
              [0,2],
              [0,1]],

    "eye_color": [[1,13,14,15,16,17,18,19,20],
                  [0,1,2,3,4,5,6,7,8],
                  ["white","black","red","green","blue","yellow","evil","purple","callus"],
                  [0,1,2,3,4,5,-1,6,-3],
                  [0,0,1,2,3,4,5,6,7]],

    "hair_color": [[1,11,16,17,18,19,20],
                   [0,1,2,3,4,5,6],
                   ["white","black","brown","blonde","red","pink","blue"],
                   [0,0,1,2,3,-1,-2],
                   [0,0,1,2,3,4,5]],

    "ears": [[1,17,18,19,20],
             [0,1,2,3,4],
             ["n/a","antannae","ears","ogre","goblin"],
             [0,1,2,3,-1],
             [0,1,2,3,4]],

    "hair_texture": [[1,20],
                     [0,1],
                     ["straight","curly"],
                     [0,2],
                     [0,1]],

    "splitting": [[1,19,20],
                  [0,1,2],
                  ["n/a","mitosion","spore"],
                  [0,0,0],
                  [0,1,2]],
}

#-----------------------------------------------------
# ROCK DEFINITION ZONE
#-----------------------------------------------------

class Sex(str, Enum):
    MALE = "male"
    FEMALE = "female"

class RockStatus(str, Enum):
    ACTIVE = "active"
    SOLD = "sold"
    DEAD = "dead"
    CRAISENED = "craisened"
    BRED = "bred"

@dataclass(frozen=True)
class Allele:
    value: int
    cost: int
    dominance: int

@dataclass(frozen=True)
class GenePair:
    allele_a: Allele
    allele_b: Allele

    @property
    def alleles(self) -> tuple[Allele, Allele]:
        return self.allele_a, self.allele_b

    # possible codminance rules here but unlikely    

@dataclass(frozen=True)
class Genome:
    genes: dict[str, GenePair] = field(default_factory=dict)

    def get(self, gene_name: str) -> GenePair:
        return self.genes[gene_name]

    @staticmethod
    def roll_gene_pair() -> list[int]:
        return [random.randint(1, 20), random.randint(1, 20)]

    @staticmethod
    def get_allele_from_roll(
        roll_value: int,
        possible_rolls: list[int],
        possible_traits: list[int],
        possible_dominance: list[str],
        possible_costs: list[int],
    ) -> Allele:
        best_match_idx = -1

        for i, roll_threshold in enumerate(possible_rolls):
            if roll_value >= roll_threshold:
                best_match_idx = i
            else:
                break

        if best_match_idx == -1:
            best_match_idx = 0

        trait_value = possible_traits[best_match_idx]
        trait_dominance = possible_dominance[best_match_idx]
        trait_cost = possible_costs[best_match_idx]

        return Allele(
            value = trait_value,
            cost = trait_cost,
            dominance = trait_dominance,
        )

    @classmethod
    def instantiate_genotype(cls, rock_roll_dict: dict) -> "Genome":
        genes: dict[str, GenePair] = {}

        # "gene": [
        #     [roll],
        #     [trait],
        #     [name],
        #     [cost],
        #     [dominance],
        # ]

        for gene_name, gene_info in rock_roll_dict.items():
            possible_rolls = gene_info[0]
            possible_traits = gene_info[1]
            possible_costs = gene_info[3]
            possible_dominance = gene_info[4]

            roll_a, roll_b = cls.roll_gene_pair()

            allele_a = cls.get_allele_from_roll(
                roll_value = roll_a,
                possible_rolls = possible_rolls,
                possible_traits = possible_traits,
                possible_dominance = possible_dominance,
                possible_costs = possible_costs,
            )

            allele_b = cls.get_allele_from_roll(
                roll_value = roll_b,
                possible_rolls = possible_rolls,
                possible_traits = possible_traits,
                possible_dominance = possible_dominance,
                possible_costs = possible_costs,
            )

            genes[gene_name] = GenePair(
                allele_a = allele_a,
                allele_b = allele_b,
            )

        return cls(genes = genes)
    

@dataclass(frozen=True)
class Phenotype:
    """
    WORK NEEDED HERE
    """
    body_color: str
    hair_color: str
    mouth_type: str
    has_horns: bool
    has_wings: bool
    size: float
    # MORE
    

@dataclass
class Rock:
    """
    ATTRIBUTES OF DE ROCK   
    """
    id: int
    name: str
    sex: Sex

    genotype: Genome = field(default_factory = Genome)
    parent_ids: list[int] = field(default_factory = list)
    generation: int = 0

    status: RockStatus = RockStatus.ACTIVE

    image_path: str | None = None
    value: int | None = None

    is_market: bool = False

    """
    METHODS OF DE ROCK   
    """
    @property
    def is_active(self) -> bool:
        return self.status == RockStatus.ACTIVE

    def change_status(self, new_status: RockStatus):
        if not isinstance(new_status, RockStatus):
            raise TypeError(f"new_status must be a RockStatus, not {type(new_status)}")
    
        self.status = new_status

    def handle_mitosion(self, game):
        pass

    def handle_sporing(self, game):
        pass

    def determine_phenotype(self):
        pass



    





















