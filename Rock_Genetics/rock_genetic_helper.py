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

@dataclass(frozen=True)
class TraitOption:
    """
    One allele-level option for a gene.
    """
    allele: int
    roll_threshold: int
    name: str
    cost: int
    dominance: int


@dataclass(frozen=True)
class PhenotypeState:
    """
    One expressed phenotype state.

    For many genes, the state key is just the allele value.
    For dosage genes, the state key may be:
        0 = none
        1 = one active copy
        2 = two active copies
    """
    key: int
    name: str
    cost: int


@dataclass(frozen=True)
class GeneSpec:
    """
    Complete specification for one gene.
    """
    name: str
    expression_rule: str
    options: dict[int, TraitOption]
    states: dict[int, PhenotypeState] = field(default_factory=dict)
    special_states: dict[tuple[int, int], PhenotypeState] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def option_for_allele(self, allele: int) -> TraitOption:
        return self.options[allele]

    def state_for_key(self, key: int) -> PhenotypeState:
        return self.states[key]
    
def make_gene_spec(
    *,
    name: str,
    rolls: list[int],
    alleles: list[int],
    allele_names: list[str],
    allele_costs: list[int],
    dominance: list[int],
    expression_rule: str = "dominance",
    state_keys: list[int] | None = None,
    state_names: list[str] | None = None,
    state_costs: list[int] | None = None,
    special_states: dict[tuple[int, int], tuple[str, int]] | None = None,
    metadata: dict[str, Any] | None = None,
) -> GeneSpec:
    """
    Build a GeneSpec from explicit lists.

    For ordinary dominance genes:
        state_names/state_costs can be omitted and will default to allele-level values.

    For dosage genes:
        pass separate state_keys/state_names/state_costs.

    special_states:
        maps allele pairs like (0, 1) to ("silver", 0)
    """
    if not (
        len(rolls) == len(alleles) == len(allele_names) == len(allele_costs) == len(dominance)
    ):
        raise ValueError(
            f"{name}: rolls/alleles/allele_names/allele_costs/dominance must have equal lengths."
        )

    options = {
        allele: TraitOption(
            allele=allele,
            roll_threshold=roll,
            name=allele_name,
            cost=cost,
            dominance=dom,
        )
        for roll, allele, allele_name, cost, dom in zip(
            rolls, alleles, allele_names, allele_costs, dominance
        )
    }

    if state_keys is None:
        state_keys = list(alleles)

    if state_names is None:
        state_names = list(allele_names)

    if state_costs is None:
        state_costs = list(allele_costs)

    if not (len(state_keys) == len(state_names) == len(state_costs)):
        raise ValueError(
            f"{name}: state_keys/state_names/state_costs must have equal lengths."
        )

    states = {
        key: PhenotypeState(key=key, name=state_name, cost=state_cost)
        for key, state_name, state_cost in zip(state_keys, state_names, state_costs)
    }

    built_special_states: dict[tuple[int, int], PhenotypeState] = {}
    if special_states is not None:
        for pair, (special_name, special_cost) in special_states.items():
            built_special_states[tuple(pair)] = PhenotypeState(
                key=-1,
                name=special_name,
                cost=special_cost,
            )

    return GeneSpec(
        name=name,
        expression_rule=expression_rule,
        options=options,
        states=states,
        special_states=built_special_states,
        metadata=metadata or {},
    )
    
@dataclass
class Phenotype:
    phenes: dict[str, str] = field(default_factory=dict)
    costs: dict[str, int] = field(default_factory=dict)

    def get(self, gene_name: str, default: str = "n/a") -> str:
        return self.phenes.get(gene_name, default)

    def cost_of(self, gene_name: str, default: int = 0) -> int:
        return self.costs.get(gene_name, default)
    

GENE_SPECS: dict[str, GeneSpec] = {
    "shape": make_gene_spec(
        name="shape",
        rolls=[1, 17, 18, 19, 20],
        alleles=[0, 1, 2, 3, 4],
        allele_names=["circle", "oval", "square", "triangle", "oblong"],
        allele_costs=[0, -1, 2, 4, -2],
        dominance=[0, 1, 2, 3, 4],
        expression_rule="dominance",
    ),

    "size": make_gene_spec(
        name="size",
        rolls=[1, 17, 18, 19, 20],
        alleles=[0, 1, 2, 3, 4],
        allele_names=["medium", "large", "small", "giant", "missized"],
        allele_costs=[0, 1, 2, 4, -2],
        dominance=[0, 1, 2, 3, 4],
        expression_rule="dominance",
    ),

    "color": make_gene_spec(
        name="color",
        rolls=[1, 11, 16, 17, 18, 19, 20],
        alleles=[0, 1, 2, 3, 4, 5, 6],
        allele_names=["white", "black", "brown", "red", "yellow", "blue", "patchwork"],
        allele_costs=[0, 0, 1, 2, 2, 2, -1],
        dominance=[0, 0, 1, 2, 2, 2, 3],
        expression_rule="body_color_codominance",
        special_states={
            (0, 1): ("silver", 0),
            (1, 0): ("silver", 0),
            (3, 4): ("orange", 2),
            (4, 3): ("orange", 2),
            (3, 5): ("purple", 2),
            (5, 3): ("purple", 2),
            (4, 5): ("green", 2),
            (5, 4): ("green", 2),
        },
    ),

    "eyes": make_gene_spec(
        name="eyes",
        rolls=[1, 18],
        alleles=[0, 1],
        allele_names=["inactive", "active"],
        allele_costs=[0, 1],
        dominance=[0, 0],
        expression_rule="dosage",
        state_keys=[0, 1, 2],
        state_names=["n/a", "eye", "double eye"],
        state_costs=[0, 1, 2],
    ),

    "brows": make_gene_spec(
        name="brows",
        rolls=[1, 18, 19, 20],
        alleles=[0, 1, 2, 3],
        allele_names=["n/a", "brows", "eyehair", "unibrows"],
        allele_costs=[0, 1, 2, -1],
        dominance=[0, 1, 2, 3],
        expression_rule="dominance",
    ),

    "mouths": make_gene_spec(
        name="mouths",
        rolls=[1, 17, 18, 19, 20],
        alleles=[0, 1, 2, 3, 4],
        allele_names=["n/a", "mouth", "smile", "chip", "smeagol"],
        allele_costs=[0, 1, 2, 3, -1],
        dominance=[0, 1, 2, 3, 4],
        expression_rule="dominance",
    ),

    "noses": make_gene_spec(
        name="noses",
        rolls=[1, 17, 18, 19, 20],
        alleles=[0, 1, 2, 3, 4],
        allele_names=["n/a", "nub", "honk", "holes", "concave"],
        allele_costs=[0, 1, 2, 3, -1],
        dominance=[0, 1, 2, 3, 4],
        expression_rule="dominance",
    ),

    "arms": make_gene_spec(
        name="arms",
        rolls=[1, 19, 20],
        alleles=[0, 1, 2],
        allele_names=["n/a", "arms", "muscle arms"],
        allele_costs=[0, 1, 2],
        dominance=[0, 0, 0],
        expression_rule="arms_special",
    ),

    "crowns": make_gene_spec(
        name="crowns",
        rolls=[1, 17, 18, 19, 20],
        alleles=[0, 1, 2, 3, 4],
        allele_names=["n/a", "small", "medium", "large", "indent"],
        allele_costs=[0, 1, 2, 3, -1],
        dominance=[0, 1, 2, 3, 4],
        expression_rule="dominance",
    ),

    "wings": make_gene_spec(
        name="wings",
        rolls=[1, 20],
        alleles=[0, 1],
        allele_names=["n/a", "wings"],
        allele_costs=[0, 2],
        dominance=[0, 1],
        expression_rule="dominance",
    ),

    "halos": make_gene_spec(
        name="halos",
        rolls=[1, 20],
        alleles=[0, 1],
        allele_names=["n/a", "halos"],
        allele_costs=[0, 2],
        dominance=[0, 1],
        expression_rule="dominance",
    ),

    "horns": make_gene_spec(
        name="horns",
        rolls=[1, 20],
        alleles=[0, 1],
        allele_names=["n/a", "horns"],
        allele_costs=[0, 2],
        dominance=[0, 1],
        expression_rule="dominance",
    ),

    "wrinkles": make_gene_spec(
        name="wrinkles",
        rolls=[1, 20],
        alleles=[0, 1],
        allele_names=["n/a", "wrinkles"],
        allele_costs=[0, 2],
        dominance=[0, 1],
        expression_rule="dominance",
    ),

    "fuzz": make_gene_spec(
        name="fuzz",
        rolls=[1, 20],
        alleles=[0, 1],
        allele_names=["inactive", "active"],
        allele_costs=[0, 1],
        dominance=[0, 1],
        expression_rule="dosage",
        state_keys=[0, 1, 2],
        state_names=["n/a", "fuzzy", "spiky"],
        state_costs=[0, 1, 2],
    ),

    "hair": make_gene_spec(
        name="hair",
        rolls=[1, 18],
        alleles=[0, 1],
        allele_names=["inactive", "active"],
        allele_costs=[0, 1],
        dominance=[0, 1],
        expression_rule="dosage",
        state_keys=[0, 1, 2],
        state_names=["n/a", "hair", "double hair"],
        state_costs=[0, 1, 2],
    ),

    "facial_hair": make_gene_spec(
        name="facial_hair",
        rolls=[0, 15, 16, 17, 18, 19, 20],
        alleles=[0, 1, 2, 3, 4, 5, 6],
        allele_names=["n/a", "goatee", "beard", "pedo", "curly", "chapman", "sol"],
        allele_costs=[0, 1, 2, -1, 3, 4, -2],
        dominance=[0, 1, 2, 3, 4, 5, 6],
        expression_rule="dominance",
    ),

    "freckles": make_gene_spec(
        name="freckles",
        rolls=[1, 20],
        alleles=[0, 1],
        allele_names=["n/a", "freckles"],
        allele_costs=[0, 2],
        dominance=[0, 1],
        expression_rule="dominance",
    ),

    "stones": make_gene_spec(
        name="stones",
        rolls=[1, 20],
        alleles=[0, 1],
        allele_names=["n/a", "stones"],
        allele_costs=[0, 2],
        dominance=[0, 1],
        expression_rule="dominance",
    ),

    "tails": make_gene_spec(
        name="tails",
        rolls=[1, 20],
        alleles=[0, 1],
        allele_names=["n/a", "tails"],
        allele_costs=[0, 2],
        dominance=[0, 1],
        expression_rule="dominance",
    ),

    "eye_color": make_gene_spec(
        name="eye_color",
        rolls=[1, 13, 14, 15, 16, 17, 18, 19, 20],
        alleles=[0, 1, 2, 3, 4, 5, 6, 7, 8],
        allele_names=["white", "black", "red", "green", "blue", "yellow", "evil", "purple", "callus"],
        allele_costs=[0, 1, 2, 3, 4, 5, -1, 6, -3],
        dominance=[0, 0, 1, 2, 3, 4, 5, 6, 7],
        expression_rule="dominance",
    ),

    "hair_color": make_gene_spec(
        name="hair_color",
        rolls=[1, 11, 16, 17, 18, 19, 20],
        alleles=[0, 1, 2, 3, 4, 5, 6],
        allele_names=["white", "black", "brown", "blonde", "red", "pink", "blue"],
        allele_costs=[0, 0, 1, 2, 3, -1, -2],
        dominance=[0, 0, 1, 2, 3, 4, 5],
        expression_rule="hair_color_codominance",
        special_states={
            (0, 1): ("silver", 0),
            (1, 0): ("silver", 0),
        },
    ),

    "ears": make_gene_spec(
        name="ears",
        rolls=[1, 17, 18, 19, 20],
        alleles=[0, 1, 2, 3, 4],
        allele_names=["n/a", "antannae", "ears", "ogre", "goblin"],
        allele_costs=[0, 1, 2, 3, -1],
        dominance=[0, 1, 2, 3, 4],
        expression_rule="dominance",
    ),

    "hair_texture": make_gene_spec(
        name="hair_texture",
        rolls=[1, 20],
        alleles=[0, 1],
        allele_names=["straight", "curly"],
        allele_costs=[0, 2],
        dominance=[0, 1],
        expression_rule="dominance",
    ),

    "splitting": make_gene_spec(
        name="splitting",
        rolls=[1, 19, 20],
        alleles=[0, 1, 2],
        allele_names=["n/a", "mitosion", "spore"],
        allele_costs=[0, 0, 0],
        dominance=[0, 1, 2],
        expression_rule="dominance",
    ),
}


    
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
        possible_dominance: list[int],
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

    phenotype: Phenotype = field(default_factory = Phenotype)
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



    





















