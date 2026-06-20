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
    money_value: int | None = None
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
 
@dataclass()
class GenomeFactory: 
    
    genome_spec_list = GENE_SPECS

    @staticmethod
    def mutate_allele(
        gene: GeneSpec,
        allele_passed: Allele,
        mutation_chance = 1/100,
    ) -> Allele:
        
        """
        IMPORTANT SETUP FOR GAME FORMAT AND MUTATION CHANCES!
        """
        if random.random() >= mutation_chance:
            return allele_passed

        possible_values = [
            gene.options[option].allele
            for option in gene.options
            if gene.options[option].allele != allele_passed.value
        ]

        if not possible_values:
            return allele_passed

        return Allele(
            value=random.choice(possible_values)
            )
        
    @staticmethod
    def roll_gene_pair() -> list[int]:
        return [random.randint(1, 20), random.randint(1, 20)]

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

    def make_random_rock_genome(
        self,
        **kwargs
    ) -> Genome:
        
        genes: dict[str, GenePair] = {}

        genome_spec_list = self.genome_spec_list

        # MAKE TOTALLY RANDOM GENOME PATH
        for gene in genome_spec_list:

            rand_genes = GenomeFactory.roll_gene_pair()

            rand_alleles = (
                GenomeFactory.get_allele_from_roll(
                    roll_value = rand_genes[0],
                    gene = genome_spec_list[gene]),
                GenomeFactory.get_allele_from_roll(
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

    def make_child_rock_genome_from_parents(
        self,
        parent_a = None,
        parent_b = None,
        **kwargs
    ) -> Genome:
            # MAKE GENOME DEPENDING ON PARENTS

            genes: dict[str, GenePair] = {}

            genome_spec_list = self.genome_spec_list

            for gene in genome_spec_list:

                if random.random() < 0.5:
                    parent_a_allele = parent_a.genotype.genes[gene].allele_a
                else:
                    parent_a_allele = parent_a.genotype.genes[gene].allele_b
                if random.random() < 0.5:
                    parent_b_allele = parent_b.genotype.genes[gene].allele_a
                else:
                    parent_b_allele = parent_b.genotype.genes[gene].allele_b
                
                parent_a_allele = GenomeFactory.mutate_allele(
                    gene = genome_spec_list[gene],
                    allele_passed= parent_a_allele,
                    **kwargs
                )

                parent_b_allele = GenomeFactory.mutate_allele(
                    gene = genome_spec_list[gene],
                    allele_passed= parent_b_allele,
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
    name: str
    sex: Sex

    genotype: Genome = field(default_factory = Genome)
    parent_ids: list[int] = field(default_factory = list)
    generation: int = 0

    status: RockStatus = RockStatus.ACTIVE

    #phenotype: Phenotype = field(default_factory = Phenotype)
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

    def instantiate_phenotype(
        self,
        rock,
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

            elif Gener.expression_rule == "dominance":
                phenotype, money =  (ExpressionEngine.dominance_phenotype_finding(
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
                    phenotype, money = (ExpressionEngine.dominance_phenotype_finding(
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

    def set_rock_value(
            self,
            rock, 
        ) -> Rock:

        genome_spec_list = self.genome_spec_list

        if rock.status == RockStatus.ACTIVE:

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

        else:
            rock.value = 0

        return(rock)
    




    





















