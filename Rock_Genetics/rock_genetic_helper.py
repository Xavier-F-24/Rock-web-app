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

from __future__ import annotations
import random

from dataclasses import dataclass, field
from enum import Enum
from typing import  Any

#-----------------------------------------------------
# ROCK GEOME ZONE
#-----------------------------------------------------

class Sex(str, Enum):
    MALE = "male"
    FEMALE = "female"

@dataclass(frozen=True)
class TraitOption:
    """
    One allele - level option for a gene
    """
    allele: int
    roll_threshold: int
    name: str
    cost: int
    dominance: int

@dataclass(frozen=True)
class PhenotypeState:
    """
    One expressed phenotype state

    For most genes, the state key is just the allele value
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
    states: dict[int, PhenotypeState] = field(default_factory = dict)
    special_states: dict[tuple[int, int], PhenotypeState] = field(default_factory = dict)
    
    required_states: dict[str, str] = field(default_factory=dict)
    required_gender: Sex | None= None
    required_gender_states: str | None = None

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

    required_states: dict[str, str] | None = None,
    required_gender: Sex | None = None,
    required_gender_states: str | None = None,

    metadata: dict[str, Any] | None = None,
    ) -> GeneSpec:
    """
    Build a GeneSpec from explicit lists

    For ordinary dominance genes:
        state_names/state_costs can be omitted and will default to allele-level values

    For dosage genes:
        pass separate state_keys/state_names/state_costs

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

    if required_states == None:
        required_states = {}

    return GeneSpec(
        name = name,
        expression_rule = expression_rule,
        options = options,
        states = states,
        special_states = built_special_states,

        required_states = required_states,
        required_gender = required_gender,
        required_gender_states = required_gender_states,

        metadata=metadata or {},
    )

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
        expression_rule="body_color_dominance",
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
        expression_rule="arms_dominance",
        special_states={
            (0, 1): ("one pair arms", 1),
            (1, 0): ("one pair arms", 1),
            (2, 0): ("one pair muscle arms", 2),
            (0, 2): ("one pair muscle arms", 2),
            (1, 2): ("arms and muscle arms", 3),
            (2, 1): ("arms and muscle arms", 3),
            (2, 2): ("two pair muscle arms", 4),
            (1, 1): ("two pair arms", 2),
        },
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
        required_gender= Sex.MALE,
        required_gender_states= "peach fuzz"
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
        required_states = {
            "eyes" : "n/a"
            }
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
        required_states = {
            "hair" : "n/a",
            "facial_hair" : "n/a",
            "brows" : "n/a"
            }
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
        required_states = {
            "hair" : "n/a",
            "facial_hair" : "n/a",
            "brows" : "n/a"
            }
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

#-----------------------------------------------------
#GENOME DEFINITION ZONE
#-----------------------------------------------------

@dataclass(frozen=True)
class Allele:
    value: int

@dataclass()
class GenePair:
    allele_a: Allele
    allele_b: Allele

    name_of_gene: str 
    dominance_type: str
    money_value: int = 0
    phenotype: str | None = None

    @property
    def alleles(self) -> tuple[Allele, Allele]:
        return self.allele_a, self.allele_b

@dataclass()
class Genome:
    genes: dict[str, GenePair] = field(default_factory=dict)

    def get_gene(self, gene_name: str) -> GenePair:
        return self.genes[gene_name]
    
    def print_genes(self):
        for gene in self.genes:
            print(f"{self.genes[gene].name_of_gene}: {self.genes[gene].allele_a.value} and {self.genes[gene].allele_b.value} as: {self.genes[gene].phenotype} worth {self.genes[gene].money_value} \n")
 
#-----------------------------------------------------
# ROCK NAME DEFINITION ZONE
#-----------------------------------------------------

@dataclass(frozen = True)
class RockName:

    given: str
    family: str | None = None
    honorific: str | None = None
    epithet: str | None = None

    #-----------------------------------------------------
    # PRINT A STR OF THE ROCKS FULL NAME SET!
    #-----------------------------------------------------

    @property
    def full_name(self) -> str:
        parts = []

        if self.honorific:
            parts.append(self.honorific)

        parts.append(self.given)

        if self.family:
            parts.append(self.family)

        if self.epithet:
            parts.append(self.epithet)

        return " ".join(parts)

#-----------------------------------------------------
# ROCK DEFINITION ZONE
#-----------------------------------------------------
class RockStatus(str, Enum):
    ACTIVE = "active"
    SOLD = "sold"
    DEAD = "dead"
    CRAISENED = "craisened"
    BRED = "bred"

@dataclass
class Rock:
    """
    ATTRIBUTES OF DE ROCK   
    """

    id: int
    sex: Sex
    name: RockName| None = None

    genotype: Genome = field(default_factory = Genome)
    death_genes: Genome = field(default_factory = Genome)
    parent_ids: list[int] = field(default_factory = list)
    generation: int = 0

    status: RockStatus = RockStatus.ACTIVE

    has_split: bool = False
    checked_craisen: bool = False

    death_reason: str | None = None

    #phenotype: Phenotype = field(default_factory = Phenotype)
    image_path: str | None = None

    value: int = 0

    sell_value: int = 0
    score_value: int = 0

    is_market: bool = False

    """
    METHODS OF DE ROCK   
    """
    @property
    def is_active(self) -> bool:
        return self.status == RockStatus.ACTIVE

    def change_status(
            self, 
            new_status: RockStatus
    ):
        if not isinstance(new_status, RockStatus):
            raise TypeError(f"new_status must be a RockStatus, not {type(new_status)}")
    
        self.status = new_status

#-----------------------------------------------------
# GENOTYPE MAKER DEFINITION ZONE
#-----------------------------------------------------

@dataclass()
class GenomeFactory: 
    
    genome_spec_list = GENE_SPECS
    death_gene_list = ["death_gene1", "death_gene2", "death_gene3"]

    rng: random.Random = field(default_factory = random.Random)

    #-----------------------------------------------------
    # SETUP DEFINITION ZONE TO MAKE ROCK GENOMES
    #-----------------------------------------------------

    def roll_gene_pair(
        self,
    ) -> list[int]:
        return [self.rng.randint(1, 20), self.rng.randint(1, 20)]

    @staticmethod
    def get_allele_from_roll(
        roll_value: int,
        gene: GeneSpec,
    ) -> Allele:
        
        chosen_option = gene.options[0]

        for option in gene.options:
            if roll_value >= gene.options[option].roll_threshold:
                chosen_option = gene.options[option]
            else:
                break

        return Allele(
            value=chosen_option.allele
            )
    
    def mutate_allele(
        self,
        gene: GeneSpec,
        allele_passed: Allele,
        mutation_chance,
    ) -> Allele:
        
        """
        IMPORTANT SETUP FOR GAME FORMAT AND MUTATION CHANCES!
        """
        if self.rng.random() >= mutation_chance:
            return allele_passed

        possible_values = [
            gene.options[option].allele
            for option in gene.options
            if gene.options[option].allele != allele_passed.value
        ]

        if not possible_values:
            return allele_passed

        return Allele(
            value = self.rng.choice(possible_values)
        )
    
    def roll_craisen_pair(
        self
    ) -> tuple[int, int]:
            a = self.rng.randint(1,100)
            b = self.rng.randint(1,100)
            while a == b:
                a = self.rng.randint(1,100)
                b = self.rng.randint(1,100)
            return (a, b)

    def mutate_death_allele(
        self,
        allele_passed: Allele,
        mutation_chance,
    ) -> Allele:
        
        """
        IMPORTANT SETUP FOR GAME FORMAT AND MUTATION CHANCES!
        """
        if self.rng.random() >= mutation_chance:
            return allele_passed

        new_allele, _ = self.roll_craisen_pair()

        while new_allele == allele_passed.value:
            new_allele, _ = self.roll_craisen_pair()

        return Allele(
            value = new_allele
        )
        
    #-----------------------------------------------------
    # NEW ROCK BEHAVIOR DEFINITION ZONE
    #-----------------------------------------------------

    def make_random_rock_genome(
        self,
        **kwargs
    ) -> Genome:
        
        genes: dict[str, GenePair] = {}

        genome_spec_list = self.genome_spec_list

        # MAKE TOTALLY RANDOM GENOME PATH
        for gene in genome_spec_list:

            rand_genes = self.roll_gene_pair()

            rand_alleles = (
                self.get_allele_from_roll(
                    roll_value = rand_genes[0],
                    gene = genome_spec_list[gene]),
                self.get_allele_from_roll(
                    roll_value = rand_genes[1],
                    gene = genome_spec_list[gene])
            )

            rand_gene_pair = GenePair(
                allele_a = rand_alleles[0],
                allele_b = rand_alleles[1],
                name_of_gene = gene,
                dominance_type = genome_spec_list[gene].expression_rule,
            )
                
            genes[gene] = rand_gene_pair

        return (Genome(genes))

    def make_selected_rock_genome(
        self,
        selected_traits: dict[str, object] | None = None,
        random_fill: bool = True,
    ) -> Genome:
        """
        Build a genome with explicit allele pairs for selected genes.

        selected_traits accepts values like:
        - {"color": (3, 4)}
        - {"eyes": [1, 1]}
        - {"shape": "22"}
        Unspecified genes are random by default, or zeroed when random_fill=False.
        """
        selected_traits = selected_traits or {}

        if random_fill:
            genome = self.make_random_rock_genome()
        else:
            genes: dict[str, GenePair] = {}
            for gene_name, spec in self.genome_spec_list.items():
                genes[gene_name] = GenePair(
                    allele_a=Allele(0),
                    allele_b=Allele(0),
                    name_of_gene=gene_name,
                    dominance_type=spec.expression_rule,
                )
            genome = Genome(genes)

        for gene_name, raw_pair in selected_traits.items():
            if gene_name not in self.genome_spec_list:
                raise KeyError(f"Unknown gene: {gene_name}")

            allele_a, allele_b = self.parse_selected_allele_pair(raw_pair)
            spec = self.genome_spec_list[gene_name]

            if allele_a not in spec.options:
                raise ValueError(f"{gene_name}: invalid allele {allele_a}")
            if allele_b not in spec.options:
                raise ValueError(f"{gene_name}: invalid allele {allele_b}")

            genome.genes[gene_name] = GenePair(
                allele_a=Allele(allele_a),
                allele_b=Allele(allele_b),
                name_of_gene=gene_name,
                dominance_type=spec.expression_rule,
            )

        return genome

    @staticmethod
    def parse_selected_allele_pair(raw_pair: object) -> tuple[int, int]:
        if isinstance(raw_pair, str):
            cleaned = raw_pair.strip()
            if len(cleaned) != 2 or not cleaned.isdigit():
                raise ValueError(f"Selected allele string must look like '01', not {raw_pair!r}")
            return int(cleaned[0]), int(cleaned[1])

        if isinstance(raw_pair, int):
            return raw_pair, raw_pair

        try:
            values = list(raw_pair)  # type: ignore[arg-type]
        except TypeError as exc:
            raise ValueError(f"Selected allele pair must be a string, int, or two values: {raw_pair!r}") from exc

        if len(values) != 2:
            raise ValueError(f"Selected allele pair must contain exactly two values: {raw_pair!r}")

        return int(values[0]), int(values[1])
    
    def make_death_genes(
        self,
    ) -> Genome:
        
        genes = {}
        
        for death_gene in self.death_gene_list:

            (a, b) = self.roll_craisen_pair()

            death_gene_pair = GenePair(
                allele_a = Allele(value = a),
                allele_b = Allele(value = b),
                name_of_gene = death_gene,
                dominance_type = "death_genes",
                phenotype = "n/a",
                money_value = 0,
                )

            genes[death_gene] = death_gene_pair

        return(Genome(genes= genes))

    #-----------------------------------------------------
    # INHERITANCE BEHAVIOR DEFINITION ZONE
    #-----------------------------------------------------

    def make_child_rock_genome_from_parents(
        self,
        parent_a : Rock,
        parent_b : Rock,
        mutation_chance,
        **kwargs
    ) -> Genome:
            
            # MAKE GENOME DEPENDING ON PARENTS

            genes: dict[str, GenePair] = {}

            genome_spec_list = self.genome_spec_list

            for gene in genome_spec_list:

                if self.rng.random() < 0.5:
                    parent_a_allele = parent_a.genotype.genes[gene].allele_a
                else:
                    parent_a_allele = parent_a.genotype.genes[gene].allele_b
                if self.rng.random() < 0.5:
                    parent_b_allele = parent_b.genotype.genes[gene].allele_a
                else:
                    parent_b_allele = parent_b.genotype.genes[gene].allele_b
                
                parent_a_allele = self.mutate_allele(
                    gene = genome_spec_list[gene],
                    allele_passed= parent_a_allele,
                    mutation_chance = mutation_chance,
                    **kwargs
                )

                parent_b_allele = self.mutate_allele(
                    gene = genome_spec_list[gene],
                    allele_passed= parent_b_allele,
                    mutation_chance = mutation_chance,
                    **kwargs
                )

                rand_gene_pair = GenePair(
                    allele_a = parent_a_allele,
                    allele_b = parent_b_allele,
                    name_of_gene = gene,
                    dominance_type = genome_spec_list[gene].expression_rule,
                )

                genes[gene] = rand_gene_pair

            return (Genome(genes))

    def inherit_death_genes(
        self,
        parent_a,
        parent_b,
        mutation_chance,
    ) -> Genome:
        
        genes = {}
        
        for death_gene in self.death_gene_list:

            if self.rng.random() < 0.5:
                parent_a_allele = parent_a.death_genes.genes[death_gene].allele_a
            else:
                parent_a_allele = parent_a.death_genes.genes[death_gene].allele_b
            if self.rng.random() < 0.5:
                parent_b_allele = parent_b.death_genes.genes[death_gene].allele_a
            else:
                parent_b_allele = parent_b.death_genes.genes[death_gene].allele_b
            
            parent_a_allele = self.mutate_death_allele(
                    allele_passed= parent_a_allele,
                    mutation_chance = mutation_chance,
                )
            
            parent_b_allele = self.mutate_death_allele(
                    allele_passed= parent_b_allele,
                    mutation_chance = mutation_chance,
                )

            death_gene_pair = GenePair(
                allele_a = parent_a_allele,
                allele_b = parent_b_allele,
                name_of_gene = death_gene,
                dominance_type = "death_genes",
                phenotype = "n/a",
                money_value = 0,
                )

            genes[death_gene] = death_gene_pair

        return(Genome(genes = genes))

#-----------------------------------------------------
# PHENOTYPE DEFINITION ZONE
#-----------------------------------------------------
@dataclass()
class ExpressionEngine: 

    genome_spec_list = GENE_SPECS

    @staticmethod
    def dominance_phenotype_finding(
        Gener,
        a, #allele_a.value
        b, #allele_b.value
    ) -> tuple[str, int]:
        lesser = a if a <= b else b

        phenotype = Gener.option_for_allele(lesser).name
        money = Gener.option_for_allele(lesser).cost

        return(phenotype, money)
    
    #-----------------------------------------------------
    # CALCULATE A ROCKS GENE VALUES AND PHENOTYPES
    #-----------------------------------------------------

    def instantiate_phenotype(
        self,
        rock: Rock,
    ) -> Rock:
        # GETS THE MONEY VALUE AND PHENOTYPE OF EACH GENE!
        
        genome_spec_list = self.genome_spec_list

        for gene in genome_spec_list:

            Gener = genome_spec_list[gene]

            a, b = rock.genotype.genes[gene].allele_a.value, rock.genotype.genes[gene].allele_b.value

            # FOR PURE DOMINANCE GENES

            # DEALING WITH WOMEN FOR FACIAL HAIR
            if rock.sex != Gener.required_gender and Gener == "facial_hair":
                phenotype = Gener.required_gender_states
                money = 1

                rock.genotype.genes[gene].phenotype, rock.genotype.genes[gene].money_value = phenotype, money

            # NEED SPECIAL WORK SO MITOSION, SPORE DOES NOT SHOW UP AS MITOSION!
            if Gener == "splitting":
                dose = 0

                dose += a if a == 1 else 0
                dose += 2*a if a == 2 else 0

                dose += b if b == 1 else 0
                dose += 2*b if b == 2 else 0

                if dose == 2:
                    phenotype, money = "mitosion", 0
                elif dose == 8:
                    phenotype, money = "spore", 0
                else:
                    phenotype, money = "n/a", 0

                rock.genotype.genes[gene].phenotype, rock.genotype.genes[gene].money_value = phenotype, money

            elif Gener.expression_rule == "dominance":
                phenotype, money =  (self.dominance_phenotype_finding(
                        Gener = Gener,
                        a = a,
                        b = b,
                        )
                    )
            
                rock.genotype.genes[gene].phenotype, rock.genotype.genes[gene].money_value = phenotype, money
                
            elif Gener.expression_rule == "dosage":
                dose = 0
                dose += 1 if a == 1 else 0
                dose += 1 if b == 1 else 0

                phenotype = Gener.state_for_key(dose).name
                money = Gener.state_for_key(dose).cost

                rock.genotype.genes[gene].phenotype, rock.genotype.genes[gene].money_value = phenotype, money
                
            else:
                #"hair_color_dominance" 
                #"body_color_dominance" 
                #"arms_dominance"]:

                a_dom = Gener.option_for_allele(a).dominance
                b_dom = Gener.option_for_allele(b).dominance

                # HAIR AND BODY COLOR RULES
                if  a != b and a_dom == b_dom:
                    phenotype = Gener.special_states[(a, b)].name
                    money = Gener.special_states[(a, b)].cost
                        
                    rock.genotype.genes[gene].phenotype, rock.genotype.genes[gene].money_value = phenotype, money

                # ARMS RULES
                elif a!= 0 and a == b and a_dom == b_dom and Gener.expression_rule == "arms_dominance":
                    phenotype = Gener.special_states[(a, b)].name
                    money = Gener.special_states[(a, b)].cost
                        
                    rock.genotype.genes[gene].phenotype, rock.genotype.genes[gene].money_value = phenotype, money
               
                else:
                    phenotype, money = (self.dominance_phenotype_finding(
                            Gener = Gener,
                            a = a,
                            b = b,
                            )
                        )
                
                    rock.genotype.genes[gene].phenotype, rock.genotype.genes[gene].money_value = phenotype, money

        return(rock)
    
#-----------------------------------------------------
# VALUE DEFINITION ZONE
#-----------------------------------------------------
@dataclass()
class ValueCalculator: 

    genome_spec_list = GENE_SPECS

    #-----------------------------------------------------
    # CALCULATE A ROCKS VALUES WITH STATUS
    #-----------------------------------------------------

    def set_rock_value(
            self,
            rock: Rock,
        ) -> Rock:

        genome_spec_list = self.genome_spec_list

        # THIS ROCK EXISTS BABEEE
        rock.value = 1

        for gene in rock.genotype.genes:
                
            hair_counter = 0

            if rock.sex == Sex.FEMALE and gene == "facial_hair" and rock.genotype.genes[gene].phenotype != "n/a":
                rock.value += 1

            elif genome_spec_list[gene].required_states == {}:
                rock.value += rock.genotype.genes[gene].money_value
            else:
                for req in genome_spec_list[gene].required_states:
                    if hair_counter == 0 and rock.genotype.genes[req].phenotype != genome_spec_list[gene].required_states[req]:
                        hair_counter = 1
                        rock.value += rock.genotype.genes[gene].money_value
              
        rock.value = max(1, rock.value)

        if rock.status == RockStatus.ACTIVE:
            rock.sell_value = rock.value
            rock.score_value = rock.value
        else:
            rock.sell_value = 0
            rock.score_value = 0

        return(rock)

#-----------------------------------------------------
# ROCK Name ZONE
#-----------------------------------------------------

@dataclass
class ParentNameInfo:

    honorifics: list[str] = field(default_factory=list)
    families: list[str] = field(default_factory=list)
    epithets: list[str] = field(default_factory=list)
    value_score: int = 0

@dataclass
class NameGenerator:

    rng: random.Random = field(default_factory = random.Random)

    name_bits_start: tuple[str, ...] = (
        "Grum", 
        "Peb", 
        "Bas", 
        "Quart", 
        "Moss", 
        "Igni", 
        "Crag", 
        "Glim", 
        "Obsi", 
        "Feld",
        "Boul", 
        "Gran", 
        "Slate", 
        "Flint", 
        "Marb", 
        "Dol", 
        "Shal", 
        "Cobb", 
        "Rubb", 
        "Geo",
        "Lava", 
        "Tuff", 
        "Clink", 
        "Chert", 
        "Jade", 
        "On", 
        "Garn", 
        "Opal", 
        "Mica", 
        "Coal",
    )

    name_bits_end: tuple[str, ...] = (
        "ble", 
        "ite", 
        "or", 
        "yx", 
        "stone", 
        "ling", 
        "rock", 
        "spar", 
        "gem", 
        "oid",
        "bert", 
        "burt", 
        "well", 
        "wick",
        "ford", 
        "son", 
        "ley", 
        "mond", 
        "more", 
        "kins",
        "by", 
        "lo", 
        "ton", 
        "nard", 
        "rick", 
        "frey", 
        "win", 
        "low", 
        "bel", 
        "grim",
    )

    family_bits_start: tuple[str, ...] = (
        "Ash", 
        "Crag", 
        "Deep", 
        "Gold", 
        "Grey", 
        "High", 
        "Low", 
        "Moon", 
        "Mud", 
        "Night",
        "Oak", 
        "Old", 
        "Red", 
        "River", 
        "Root", 
        "Snow", 
        "Star", 
        "Sun", 
        "Thorn", 
        "Wolf",
        "Black", 
        "Bright", 
        "Broken", 
        "Cold", 
        "Copper", 
        "Dark", 
        "Dust", 
        "Iron", 
        "Royal", 
        "Silver",
    )

    family_bits_end: tuple[str, ...] = (
        "crag", 
        "fall", 
        "field", 
        "forge", 
        "gem", 
        "grip", 
        "grove", 
        "hall", 
        "helm", 
        "hill",
        "horn", 
        "keep", 
        "maw", 
        "more", 
        "ridge", 
        "root", 
        "shard", 
        "spark", 
        "stone", 
        "vale",
        "watch", 
        "well", 
        "wick", 
        "wood", 
        "yard", 
        "crest", 
        "heart", 
        "jaw", 
        "mark", 
        "peak",
    )

    male_honorifics: tuple[str, ...] = (
        "Sir", 
        "Lord", 
        "Master", 
        "Baron", 
        "Duke", 
        "Elder", 
        "Captain", 
        "Professor",
    )

    female_honorifics: tuple[str, ...] = (
        "Lady", 
        "Dame", 
        "Madam", 
        "Baroness", 
        "Duchess", 
        "Elder", 
        "Captain", 
        "Professor",
    )

    neutral_honorifics: tuple[str, ...] = (
        "Elder", 
        "Sage", 
        "Captain", 
        "Professor", 
        "Honored", 
        "Grand", 
        "Ancient",
    )

    ordinary_epithets: tuple[str, ...] = (
        "the Rock",
        "the Small",
        "the Large",
        "the Suspicious",
        "the Unbothered",
        "the Crunchy",
        "the Polished",
        "the Questionable",
        "the Round",
        "the Slightly Damp",
        "the Ancient",
        "the Unusually Shiny",
        "the Mildly Dangerous",
        "the Patient",
        "the Wiggly",
        "the Gravelhearted",
        "the Tiny Menace",
        "the Pebbleborn",
        "the Moss-Touched",
        "the Deeply Confused",
        "the Well-Formed",
        "the Troubled",
        "the Noble",
        "the Unreasonably Valuable",
    )

    noble_epithets: tuple[str, ...] = (
        "the ROCK",
        "the Highborn",
        "the Gem-Heir",
        "the Crowned Shard",
        "of the Old Line",
        "of the Glittering House",
        "of the Deep Quarry",
        "of the First Stones",
        "of the Royal Vein",
        "of the Ancient Crag",
    )

    # Base chances
    family_name_chance: float = 0.10
    honorific_chance: float = 0.05
    epithet_chance: float = 0.05

    # Inheritance chances
    inherit_family_chance: float = 0.75
    inherit_honorific_chance: float = 0.10
    inherit_epithet_chance: float = 0.15

    #-----------------------------------------------------
    # NAME NECESSARIES
    #-----------------------------------------------------

    def get_structured_name(
        self, 
        parent: Rock | None = None,
    ) -> RockName | None:

        # Rock names
        rock_name = getattr(parent, "name", None)
        if isinstance(rock_name, RockName):
            return rock_name

        # Separate fields case
        honorific = getattr(parent, "honorific", None)
        given = getattr(parent, "given", None)
        family = getattr(parent, "family", None)
        epithet = getattr(parent, "epithet", None)

        if given:
            return RockName(
                honorific = honorific,
                given = given,
                family = family,
                epithet = epithet,
            )

        return None

    def collect_parent_name_info(
        self,
        parent_a: Rock | None = None,
        parent_b: Rock | None = None,
    ) -> ParentNameInfo:
        
        info = ParentNameInfo()

        for parent in (parent_a, parent_b):
            if parent is None:
                continue

            value = getattr(parent, "value", 0) or 0
            info.value_score += value

            rock_name = self.get_structured_name(parent)

            if rock_name is not None:
                if rock_name.honorific:
                    info.honorifics.append(rock_name.honorific)
                if rock_name.family:
                    info.families.append(rock_name.family)
                if rock_name.epithet:
                    info.epithets.append(rock_name.epithet)

        return info

    def make_given_name(self) -> str:

        start = self.rng.choice(self.name_bits_start)
        end = self.rng.choice(self.name_bits_end)

        return start + end

    def make_family_name(self) -> str:

        start = self.rng.choice(self.family_bits_start)
        end = self.rng.choice(self.family_bits_end)

        return start + end

    def choose_family_name(
        self,
        parent_info: ParentNameInfo,
        force: bool = False,
    ) -> str | None:
        
        chance = self.family_name_chance

        if parent_info.value_score >= 6:
            chance += 0.10
        if parent_info.value_score >= 8:
            chance += 0.20
        if parent_info.value_score >= 12:
            chance += 0.40
        if parent_info.value_score >= 16:
            chance += 0.60

        # If parents have a family name, child is more likely to inherit one.
        if parent_info.families:
            chance += self.inherit_family_chance

        # Rock does not hit its name chance and goes home empty handed
        if not force and self.rng.random() > chance:
            return None

        # Rock chooses from parent names, where applicable (if 1 -> 1)
        if parent_info.families and self.rng.random() < 0.90:
            return self.rng.choice(parent_info.families)

        # Rock generates a new family name for itself
        return self.make_family_name()

    def get_honorific_pool(
        self, 
        sex: Any
    ) -> tuple[str, ...]:
        
        sex_text = str(sex).lower()
        
        if self.rng.random() < 0.75:

            # This works with Sex.MALE, "male", "MALE", etc.
            if "female" in sex_text:
                return self.female_honorifics

            if "male" in sex_text:
                return self.male_honorifics

        return self.neutral_honorifics

    def choose_honorific(
        self,
        sex: Sex,
        parent_info: ParentNameInfo,
        force: bool = False,
    ) -> str | None:
        
        chance = self.honorific_chance

        if parent_info.value_score >= 6:
            chance += 0.05
        if parent_info.value_score >= 8:
            chance += 0.15
        if parent_info.value_score >= 12:
            chance += 0.25
        if parent_info.value_score >= 16:
            chance += 0.40

        # If a parent has an honorific, child has a strong chance to inherit prestige.
        if parent_info.honorifics:
            chance += self.inherit_honorific_chance

        # Rock does not hit its name chance and goes home empty handed
        if not force and self.rng.random() > chance:
            return None

        # Usually inherit the parent's honorific style if available.
        if parent_info.honorifics and self.rng.random() < 0.50:
            return self.rng.choice(parent_info.honorifics)

        # Rock makes a new honnorific for itself
        return self.rng.choice(
            self.get_honorific_pool(sex)
        )

    def choose_epithet(
        self,
        parent_info: ParentNameInfo,
        child_value: int | None = None,
        force: bool = False,
    ) -> str | None:
        
        chance = self.epithet_chance

        total_value = parent_info.value_score

        if child_value is not None:
            total_value += child_value

        if total_value >= 6:
            chance += 0.05
        if total_value >= 8:
            chance += 0.15
        if total_value >= 12:
            chance += 0.25
        if total_value >= 16:
            chance += 0.40

        # If parents have titles, child is more likely to inherit title energy.
        if parent_info.epithets:
            chance += self.inherit_epithet_chance

        # Rock does not hit its name chance and goes home empty handed
        if not force and self.rng.random() > chance:
            return None

        # Valuable lines get noble epithets more often.
        if total_value >= 10 and self.rng.random() < 0.85:
            return self.rng.choice(self.noble_epithets)
        
        # Sometimes directly inherit a parent's epithet/title.
        if parent_info.epithets and self.rng.random() < 0.40:
            return self.rng.choice(parent_info.epithets)

        return self.rng.choice(self.ordinary_epithets)

    #-----------------------------------------------------
    # MAKE THAT ROCK NAME!!!
    #-----------------------------------------------------

    def generate_name(
        self,
        sex: Sex,
        parent_a: Rock | None = None,
        parent_b: Rock | None = None,

        value: int | None = None,

        force_family: bool = False,
        force_honorific: bool = False,
        force_epithet: bool = False,
    ) -> RockName:
        
        """
        Generates a structured rock name.

        parent_a and parent_b can be Rock objects.
        This method checks for:
        - parent.name
        - parent.honorific
        - parent.family
        - parent.epithet
        - parent.value
        """

        parent_info = self.collect_parent_name_info(
            parent_a,
            parent_b,
        )

        given = self.make_given_name()

        family = self.choose_family_name(
            parent_info = parent_info,
            force = force_family,
        )

        honorific = self.choose_honorific(
            sex = sex,
            parent_info = parent_info,
            force = force_honorific,
        )

        epithet = self.choose_epithet(
            parent_info = parent_info,
            child_value = value,
            force = force_epithet,
        )

        return RockName(
            honorific = honorific,
            given = given,
            family = family,
            epithet = epithet,
        )
