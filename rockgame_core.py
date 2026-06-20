import math
import random
import base64
import io
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from matplotlib.patches import (
    Polygon, Circle, Ellipse, Arc, PathPatch
)
from matplotlib.path import Path

from matplotlib.path import Path
from matplotlib.patches import PathPatch
import matplotlib.colors as mcolors

import hashlib
import plotly.express as px

@dataclass
class Rock:
    id: int
    name: str
    genes: dict
    parents: Optional[Tuple[int, int]] = None
    generation: int = 0


DEFAULT_STARTING_MONEY = 10
DEFAULT_MAX_GENERATION = 7
DEFAULT_MAX_PAIRS_PER_GENERATION = 3

RANDOM_IMPORT_COST = 8
REQUESTED_IMPORT_BASE_COST = 8
REQUESTED_IMPORT_MULTIPLIER = 2.0

CHILD_DEATH_CHANCE = 0.05
CLUTCH_MEAN = 1.5
CLUTCH_STD = 2.0
MAX_CLUTCH_SIZE = None

SPORE_CLONE_COUNT = 4
SPORE_PUFF_CHANCE = 0.25

POTION_MUTATION_RATE = 0.12
FERTILITY_EXTRA_CHILDREN = 2
ANTI_CRAISEN_REROLLS = 3

PAD_FRAC = 0.2

@dataclass
class GameState:
    player_name: str = ""
    cabal_curse_enabled: bool = False

    rocks: Dict[int, Rock] = field(default_factory=dict)
    next_id: int = 1

    generation: int = 0
    max_generation: int = DEFAULT_MAX_GENERATION

    money: int = DEFAULT_STARTING_MONEY

    max_pairs_per_generation: int = DEFAULT_MAX_PAIRS_PER_GENERATION
    breeding_queue: List[Tuple[int, int]] = field(default_factory=list)

    potions: Dict[str, int] = field(default_factory=dict)
    events: List[str] = field(default_factory=list)

    game_over: bool = False

    market_pods: List[dict] = field(default_factory=list)
    pending_market_pod: Optional[dict] = None

"""
@dataclass
class GameState:
    rocks: Dict[int, Rock] = field(default_factory=dict)
    next_id: int = 1
    generation: int = 0
    max_generation: int = 7
    money: int = 0
    max_pairs_per_generation: int = 3
    breeding_queue: list = field(default_factory=list)
    potions: dict = field(default_factory=dict)
    events: list = field(default_factory=list)
    game_over: bool = False
"""


POTION_SHOP = {
    "anti_craisen": {
        "name": "Anti-Craisen Potion",
        "cost": 5,
        "description": "Reduce or reroll craisen offspring risk."
    },
    "mutation": {
        "name": "Mutation Potion",
        "cost": 5,
        "description": "Increase mutation chance for one breeding pair."
    },
    "fertility": {
        "name": "Fertility Potion",
        "cost": 3,
        "description": "Produce extra child from one pair."
    },
    "reroll": {
        "name": "Reroll Potion",
        "cost": 3,
        "description": "Reroll clutch size from one pair."
    },
}

GENE_NAMES = [
    #"""
    #"red",
    #"yellow",
    #"blue",
    #"white",
    #"black",
    #"eyes",
    #"mouth",
    #"crystals",
    #"horns",
    #"spots",
    #"stripes",
    #"large",
    #"rough",
    #"""

    "gender",
    #["X","Y"]
    #["0","1"]
    "shape",
    #[circle, oval, square, triangle, oblong]
    #[  "0",   "1",    "2",    "3",    "4"]
    #[    0,   -1,      2,      4,     -2]
    #[    1,   17,     18,    19,      20]
    "size",
    #[medium, large, small, giant, missized]
    #[  "0",   "1",    "2",   "3",    "4"]
    #[    0,     1,     2,     4,     -2]
    #[    1,    17,    18,    19,      20]
    "color",
    #[W/B, brown, red/blue/yellow, patchwork]
    #["0","1","2",  "3","4","5",      "6"]
    #[  0, 0,  1,    2,  2,  2,      -1]
    #[  1, 11, 16,   17, 18,  19,      20]
    "eyes",
    #[no, yes]
    #["0","1"]
    #[ 0, 1]
    #[ 1, 18]
    "brows",
    #[no, brows, eyehair, unibrows]
    #["0", "1",   "2",       "3"]
    #[ 0 ,  1,     2,       -1]
    #[ 1,  18,   19,     20]
    "mouths",
    #[no, mouth, smile, chip, smeagol]
    #["0", "1",   "2",   "3",   "4"]
    #[  0,  1,     2 ,     3,   -1]
    #[  1, 17,    18 ,    19,   20]
    "noses",
    #[no, nub, honk, holes, concave]
    #["0","1", "2",   "3",    "4"]
    #[ 0 , 1 ,  2 ,    3 ,     -1]
    #[ 1 , 17, 18 ,   19 ,     20]
    "arms",
    #[no, arm, musclearm]
    #["0", "1",  "2"]
    #[ 0 ,  1 ,   2 ]
    #[ 1 , 19 ,  20 ]
    "crowns",
    #[no, small, med, big, indent]
    #["0", "1",  "2",  "3", "4"]
    #[ 0 ,  1 ,   2 ,   3 ,  -1]
    #[ 1 , 17 ,  18 ,  19 ,  20]
    "wings",
    #[no, yes]
    #["0", "1"]
    #[ 0,  2 ]
    #[ 1,  20 ]
    "halos",
    #[no, yes]
    #["0", "1"]
    #[ 0,  2 ]
    #[ 1,  20 ]
    "horns",
    #[no, yes]
    #["0", "1"]
    #[ 0,  2 ]
    #[ 1,  20 ]
    "wrinkles",
    #[no, yes]
    #["0", "1"]
    #[ 0,  2 ]
    #[ 1,  20 ]
    "fuzz",
    #[no, yes]
    #["0", "1"]
    #[ 0,  2 ]
    #[ 1,  20 ]
    "hair",
    #[no, yes]
    #["0","1"]
    #[ 0, 1]
    #[ 1, 18]
    "facial_hair",
    #[no, goatee, beard, pedo, curl, hit, sol]
    #["0", "1",    "2",   "3",  "4",  "5", "6"]
    #[ 0 ,  1 ,     2 ,    -1,   3 ,   4 ,  -2]
    "freckles",
    #[no, yes]
    #["0", "1"]
    #[ 0,  2 ]
    #[ 1,  20 ]
    "stones",
    #[no, yes]
    #["0", "1"]
    #[ 0,  2 ]
    #[ 1,  20 ]
    "tails",
    #[no, yes]
    #["0", "1"]
    #[ 0,  2 ]
    #[ 1,  20 ]
    "eye_color",
    #[W, Black, red, green, blue, yellow, evil, purple, callus]
    #["0", "1", "2",  "3",  "4",    "5",   "6",   "7",    "8"]
    #[ 0 ,  1 ,  2 ,   3 ,   4 ,     5 ,    -1,    6 ,     -3]
    #[ 1 , 13 ,  14 ,  15 ,  16 ,    17 ,    18,   19 ,    20]
    "hair_color",
    #[W/B, brown, blonde, red, pink, blue]
    #["0","1","2",  "3",  "4",  "5", "6"]
    #[ 0,  0 , 1 ,   2 ,   3 ,   -1, -2]
    #[ 1,  11, 16 ,  17 ,   18 ,   19, 20]
    "ears",
    #[no, antannae, ears, ogre, goblin]
    #["0",   "1",    "2",  "3",  "4"]
    #[ 0 ,    1 ,     2 ,   3 ,   -1]
    #[ 1 ,   17 ,    18 ,  19 ,   20]
    "hair_texture",
    #[straight, curly]
    #[  "0",     "1"]
    #[   0 ,      2]
    #[    1,     20 ]
    "splitting",
    #[no, mitosion, spore]
    #["0",   "1",    "2"]
    #[ 1 ,   19,    20]
    "death_gene1",
    #[0-100]
    # Remember to check when generating if a rock has craisen - none of that!
    "death_gene2",
    #[0-100]
    "death_gene3"
    #[0-100]
]

Rock_roll_dict = { # "gene": [[roll],[trait],[name],[cost]]
    "shape": [[1,17,18,19,20],[0,1,2,3,4],["circle","oval","square","triangle","oblong"],[0,-1,2,4,-2]],
    "size": [[1,17,18,19,20],[0,1,2,3,4],["medium","large","small","giant","missized"],[0,1,2,4,-2]],
    "color": [[1,11,16,17,18,19,20],[0,1,2,3,4,5,6],["white","black","brown","red","yellow","blue","patchwork"],[0,0,1,2,2,2,-1]],
    "eyes": [[1,18],[0,1],["n/a","eye","double eye"],[0,1,2]],
    "brows": [[1,18,19,20],[0,1,2,3],["n/a","brows","eyehair","unibrows"],[0,1,2,-1]],
    "mouths": [[1,17,18,19,20],[0,1,2,3,4],["n/a","mouth","smile","chip","smeagol"],[0,1,2,3,-1]],
    "noses": [[1,17,18,19,20],[0,1,2,3,4],["n/a","nub","honk","holes","concave"],[0,1,2,3,-1]],
    "arms": [[1,19,20],[0,1,2],["n/a","arms","muscle arms"],[0,1,2]],
    "crowns": [[1,17,18,19,20],[0,1,2,3,4],["n/a","small","medium","large","indent"],[0,1,2,3,-1]],
    "wings": [[1,20],[0,1],["n/a","wings"],[0,2]],
    "halos": [[1,20],[0,1],["n/a","halos"],[0,2]],
    "horns": [[1,20],[0,1],["n/a","horns"],[0,2]],
    "wrinkles": [[1,20],[0,1],["n/a","wrinkles"],[0,2]],
    "fuzz": [[1,20],[0,1],["n/a","fuzzy","spiky"],[0,1,2]],
    "hair": [[1,18],[0,1],["n/a","hair","double hair"],[0,1,2]],
    "facial_hair": [[0,15,16,17,18,19,20],[0,1,2,3,4,5,6],["n/a","goatee","beard","pedo","curly","chapman","sol"],[0,1,2,-1,3,4,-2]],
    "freckles": [[1,20],[0,1],["n/a","freckles"],[0,2]],
    "stones": [[1,20],[0,1],["n/a","stones"],[0,2]],
    "tails": [[1,20],[0,1],["n/a","tails"],[0,2]],
    "eye_color":[[1,13,14,15,16,17,18,19,20],[0,1,2,3,4,5,6,7,8],["white","black","red","green","blue","yellow","evil","purple","callus"],[0,1,2,3,4,5,-1,6,-3]],
    "hair_color": [[1,11,16,17,18,19,20],[0,1,2,3,4,5,6],["white","black","brown","blonde","red","pink","blue"],[0,0,1,2,3,-1,-2]],
    "ears": [[1,17,18,19,20],[0,1,2,3,4],["n/a","antannae","ears","ogre","goblin"],[0,1,2,3,-1]],
    "hair_texture": [[1,20],[0,1],["straight","curly"],[0,2]],
    "splitting": [[1,19,20],[0,1,2],["n/a","mitosion","spore"],[0,0,0]],
}

DEATH_GENES = ["death_gene1", "death_gene2", "death_gene3"]

REQUEST_TRAIT_CATALOG = {
    "gender": {
        "label": "Gender",
        "options": {
            "Female": "00",
            "Male": "01",
        },
    },

    "shape": {
        "label": "Shape",
        "options": {
            "Circle": "00",
            "Oval": "11",
            "Square": "22",
            "Triangle": "33",
            "Oblong": "44",
        },
    },

    "size": {
        "label": "Size",
        "options": {
            "Medium": "00",
            "Large": "11",
            "Small": "22",
            "Giant": "33",
            "Missized": "44",
        },
    },

    "color": {
        "label": "Body Color",
        "options": {
            "White": "00",
            "Black": "11",
            "Silver": "01",
            "Brown": "22",
            "Red": "33",
            "Yellow": "44",
            "Blue": "55",
            "Orange": "34",
            "Purple": "35",
            "Green": "45",
            "Patchwork": "66",
        },
    },

    "eyes": {
        "label": "Eyes",
        "options": {
            "No Eyes": "00",
            "One Eye": "01",
            "Two Eyes": "11",
        },
    },

    "eye_color": {
        "label": "Eye Color",
        "options": {
            "White": "00",
            "Black": "11",
            "Red": "22",
            "Green": "33",
            "Blue": "44",
            "Yellow": "55",
            "Evil": "66",
            "Purple": "77",
            "Callus": "88",
        },
    },

    "brows": {
        "label": "Brows",
        "options": {
            "No Brows": "00",
            "Brows": "11",
            "Eyelash": "22",
            "Unibrow": "33",
        },
    },

    "mouths": {
        "label": "Mouth",
        "options": {
            "No Mouth": "00",
            "Mouth": "11",
            "Smile": "22",
            "Chip": "33",
            "Smeagol": "44",
        },
    },

    "noses": {
        "label": "Nose",
        "options": {
            "No Nose": "00",
            "Nub": "11",
            "Honk": "22",
            "Holes": "33",
            "Concave": "44",
        },
    },

    "arms": {
        "label": "Arms",
        "options": {
            "No Arms": "00",
            "Normal Pair": "01",
            "Double Normal": "11",
            "Muscle Pair": "02",
            "Mixed Normal/Muscle": "12",
            "Double Muscle": "22",
        },
    },

    "crowns": {
        "label": "Crown",
        "options": {
            "No Crown": "00",
            "Small Crown": "11",
            "Medium Crown": "22",
            "Large Crown": "33",
            "Indent": "44",
        },
    },

    "wings": {
        "label": "Wings",
        "options": {
            "No Wings": "00",
            "Wings": "11",
        },
    },

    "halos": {
        "label": "Halo",
        "options": {
            "No Halo": "00",
            "Halo": "11",
        },
    },

    "horns": {
        "label": "Horns",
        "options": {
            "No Horns": "00",
            "Horns": "11",
        },
    },

    "wrinkles": {
        "label": "Wrinkles",
        "options": {
            "No Wrinkles": "00",
            "Wrinkles": "11",
        },
    },

    "fuzz": {
        "label": "Fuzz",
        "options": {
            "No Fuzz": "00",
            "Fuzz": "01",
            "Spiky": "11",
        },
    },

    "freckles": {
        "label": "Freckles",
        "options": {
            "No Freckles": "00",
            "Freckles": "11",
        },
    },

    "stones": {
        "label": "Ion Stone",
        "options": {
            "No Stone": "00",
            "Ion Stone": "11",
        },
    },

    "tails": {
        "label": "Tail",
        "options": {
            "No Tail": "00",
            "Tail": "11",
        },
    },

    "ears": {
        "label": "Ears",
        "options": {
            "No Ears": "00",
            "Antannae": "11",
            "Human": "22",
            "Ogre": "33",
            "Goblin": "44",
        },
    },

    "facial_hair": {
        "label": "Facial Hair",
        "options": {
            "None": "00",
            "Goatee": "11",
            "Beard": "22",
            "Pedo": "33",
            "Curly": "44",
            "Chapman": "55",
            "Sol": "66",
        },
    },

    "hair": {
        "label": "Hair",
        "options": {
            "No Hair": "00",
            "Hair": "01",
            "Double Hair": "11",
        },
    },

    "hair_color": {
        "label": "Hair Color",
        "options": {
            "White": "00",
            "Black": "11",
            "Silver": "01",
            "Brown": "22",
            "Blonde": "33",
            "Red": "44",
            "Pink": "55",
            "Blue": "66",
        },
    },

    "hair_texture": {
        "label": "Hair Texture",
        "options": {
            "Straight": "00",
            "Curly": "11",
        },
    },

    "splitting": {
        "label": "Splitting",
        "options": {
            "None": "00",
            "Mitosion": "11",
            "Spore": "22",
        },
    },
}

REQUEST_FORCE_00_IF_UNCHECKED = {
    "eyes",
    "hair",
    "arms",
    "fuzz",
}

COLOR_IF_UNCHECKED = {
    "hair_color",
    "color",
}

REQUEST_TRAIT_DEPENDENCIES = {
    "eye_color": {
        "parent": "eyes",
        "parent_label": "Eyes",
        "blocked_parent_values": {"00"},   # No Eyes
        "message": "Requires Eyes"
    },

    "hair_color": {
        "parent": "hair",
        "parent_label": "Hair",
        "blocked_parent_values": {"00"},   # No Hair
        "message": "Requires Hair"
    },

    "hair_texture": {
        "parent": "hair",
        "parent_label": "Hair",
        "blocked_parent_values": {"00"},   # No Hair
        "message": "Requires Hair"
    },
}

def random_rock_name() -> str:
    return random.choice(name_bits_start) + random.choice(name_bits_end)

def roll_gene_pair() -> List[int]:
    return [random.randint(1,20), random.randint(1,20)]

def roll_gender_pair() -> List[int]:
    return f"0{random.choice('01')}"

def roll_craisen_pair() -> List[int]:
    a = random.randint(1,99)
    b = random.randint(1,99)
    if a == b:
        return roll_craisen_pair()
    return f"{a:02d}{b:02d}"

def get_trait_from_roll(roll_value: int, possible_rolls: List[int], possible_traits: List[int]) -> int:
    best_match_idx = -1
    for i, r_threshold in enumerate(possible_rolls):
        if roll_value >= r_threshold:
            best_match_idx = i
        else:
            break
    if best_match_idx == -1:
        return possible_traits[0]
    return possible_traits[best_match_idx]

def make_rock_genome() -> Dict[str, str]:
    genome = {}
    for gene in GENE_NAMES:
        if gene in Rock_roll_dict:
            roll_values = roll_gene_pair() # Returns [int, int]
            gene_info = Rock_roll_dict[gene]
            possible_rolls = gene_info[0]
            possible_traits = gene_info[1]

            trait1 = get_trait_from_roll(roll_values[0], possible_rolls, possible_traits)
            trait2 = get_trait_from_roll(roll_values[1], possible_rolls, possible_traits)
            genome[gene] = f"{trait1}{trait2}"
        elif gene == "death_gene1" or gene == "death_gene2" or gene == "death_gene3":
            genome[gene] = str(roll_craisen_pair())
        else:
            # For genes not in Rock_roll_dict, use the original binary pair logic
            genome[gene] = str(roll_gender_pair())
    return genome

def make_random_rock(rock_id: int, generation: int = 0) -> Rock:
    return Rock(
        id=rock_id,
        name=random_rock_name(),
        genes=make_rock_genome(),
        parents=None,
        generation=generation
    )

def clone_with_gene(base_rock, new_id, gene_name, gene_value, name=None):
    new_genes = dict(base_rock.genes)
    new_genes[gene_name] = gene_value

    return Rock(
        id=new_id,
        name=name if name is not None else f"Test{new_id}",
        genes=new_genes,
        parents=None,
        generation=0
    )

def get_rock_phenotype(rock: Rock) -> Dict[str, str]:
    phenotype = {}
    #print(rock.rock_cost)
    for gene_name in GENE_NAMES:
        gene_pair_value = rock.genes.get(gene_name)

        if gene_pair_value is None:
            phenotype[gene_name] = "N/A (gene not present)"
            continue

        if gene_name in Rock_roll_dict:
            trait_names = Rock_roll_dict[gene_name][2]

            if gene_name == "arms":
                c1 = gene_pair_value.count('1')
                c2 = gene_pair_value.count('2')
                if c1 == 0 and c2 == 0:
                    phenotype[gene_name] = "n/a"
                else:
                  if c1 > 0:
                    army = f"{2*c1} arms"
                  else:
                    army = ""
                  if c2 > 0:
                    army2 = f"{2*c2} muscle arms"
                  else:
                    army2 = ""
                  if c1 > 0 and c2 > 0:
                    middle = ", "
                  else:
                    middle = ""

                    phenotype[gene_name] = f"{army}{middle}{army2}"
                    #print(f"{gene_name}: {c1 * int(Rock_roll_dict[gene_name][3][1]) + c2 * int(Rock_roll_dict[gene_name][3][2])}")
                    rock.rock_cost += (c1 * int(Rock_roll_dict[gene_name][3][1]) + c2 * int(Rock_roll_dict[gene_name][3][2]))

            elif gene_name in ["eyes", "fuzz", "hair"]:
                expressed_idx = np.count_nonzero([int(d) for d in gene_pair_value if d.isdigit()])
                phenotype[gene_name] = trait_names[expressed_idx]
                #print(f"{gene_name}: {int(Rock_roll_dict[gene_name][3][expressed_idx])}")
                rock.rock_cost += int(Rock_roll_dict[gene_name][3][expressed_idx])
            else:
                try:
                    expressed_idx = int(min(gene_pair_value))
                    phenotype[gene_name] = trait_names[expressed_idx]
                    if gene_name not in ["facial_hair", "eye_color", "hair_color", "hair_texture"]:
                      #print(f"{gene_name}: {int(Rock_roll_dict[gene_name][3][expressed_idx])}")
                      rock.rock_cost += int(Rock_roll_dict[gene_name][3][expressed_idx])
                except (ValueError, IndexError):
                    phenotype[gene_name] = f"Error interpreting gene '{gene_name}': '{gene_pair_value}'"

        elif gene_name in ["death_gene1", "death_gene2", "death_gene3"]:
            rock.is_craisen = 1 if f"{gene_pair_value[0]}{gene_pair_value[1]}" == f"{gene_pair_value[2]}{gene_pair_value[3]}" else 0

        elif gene_name == "gender":
            phenotype[gene_name] = "Male" if gene_pair_value == "01" else "Female"
            rock.gender = 1 if gene_pair_value == "01" else 0

        else:
            phenotype[gene_name] = gene_pair_value

    if rock.gender == 1:
      #print(f"facial hair: {int(Rock_roll_dict["facial_hair"][3][int(min(rock.genes.get("facial_hair")))])}")
      rock.rock_cost += int(Rock_roll_dict["facial_hair"][3][int(min(rock.genes.get("facial_hair")))])
    elif phenotype["facial_hair"] != "n/a":
      phenotype["facial_hair"] = "peach fuzz"
      #print(f"the fuzz: {rock.rock_cost}")
      rock.rock_cost += 1

    if phenotype["facial_hair"] != "n/a" or phenotype["hair"] != "n/a" or phenotype["brows"] != "n/a":
      #print(f"hair color: {int(Rock_roll_dict["hair_color"][3][int(min(rock.genes.get("hair_color")))])}")
      rock.rock_cost += int(Rock_roll_dict["hair_color"][3][int(min(rock.genes.get("hair_color")))])
      #print(f"hair texture: {int(Rock_roll_dict["hair_texture"][3][int(min(rock.genes.get("hair_texture")))])}")
      rock.rock_cost += int(Rock_roll_dict["hair_texture"][3][int(min(rock.genes.get("hair_texture")))])

    if phenotype["eyes"] != "n/a":
      #print(f"eye color: {int(Rock_roll_dict["eye_color"][3][int(min(rock.genes.get("eye_color")))])}")
      rock.rock_cost += int(Rock_roll_dict["eye_color"][3][int(min(rock.genes.get("eye_color")))])

    rock.rock_cost += 1 #THE ROCK EXISTS!

    #print(rock.rock_cost)
    return phenotype

def get_pair_values(rock, gene_name):
    """
    For normal categorical genes like '34', return [3, 4].
    For missing/bad genes, return [].
    """
    raw = str(rock.genes.get(gene_name, ""))

    values = []
    for ch in raw:
        if ch.isdigit():
            values.append(int(ch))

    return values

def trait_name_from_value(gene_name, value):
    """
    Convert trait index to trait name using Rock_roll_dict.
    """
    if gene_name not in Rock_roll_dict:
        return str(value)

    trait_names = Rock_roll_dict[gene_name][2]

    if 0 <= value < len(trait_names):
        return trait_names[value]

    return f"trait_{value}"

def get_trait_allele_names(rock, gene_name):
    values = get_pair_values(rock, gene_name)
    return [trait_name_from_value(gene_name, v) for v in values]

def count_nonzero_alleles(rock, gene_name):
    values = get_pair_values(rock, gene_name)
    return sum(1 for v in values if v != 0)

def get_primary_trait_name(rock, gene_name):
    """
    Uses your current dominance rule:
    smaller trait index wins for most categorical genes.
    """
    values = get_pair_values(rock, gene_name)

    if len(values) == 0:
        return "n/a"

    expressed_idx = min(values)
    return trait_name_from_value(gene_name, expressed_idx)

def check_craisen(rock):
    """
    A rock is craisen if ANY death gene has matching paired values.
    Example: '1717' means death hit.
    """
    for gene_name in DEATH_GENES:
        raw = str(rock.genes.get(gene_name, "")).zfill(4)

        if len(raw) >= 4:
            left = raw[:2]
            right = raw[2:4]

            if left == right and random.random() < 0.5:
                return True

    if rock.is_craisen == 1:
        rock.rock_cost *= 0

    return False

EYE_COLOR_MAP = {
    "white":  "white",
    "black":  "black",
    "red":    "red",
    "green":  "green",
    "blue":   "royalblue",
    "yellow": "gold",
    "evil":   "crimson",
    "purple": "purple",
    "callus": "tan",
    "n/a":    "black",
}
# -----------------------------
# Body color system
# -----------------------------

BODY_COLOR_MAP = {
    "white":     (0.88, 0.86, 0.80),
    "black":     (0.16, 0.15, 0.15),
    "silver":    (0.62, 0.62, 0.58),

    "brown":     (0.48, 0.30, 0.18),

    "red":       (0.74, 0.25, 0.22),
    "yellow":    (0.90, 0.72, 0.25),
    "blue":      (0.24, 0.40, 0.72),

    "orange":    (0.92, 0.46, 0.16),
    "green":     (0.28, 0.62, 0.30),
    "purple":    (0.52, 0.25, 0.65),

    "patchwork": (0.52, 0.48, 0.42),
    "n/a":       (0.45, 0.42, 0.38),
}

BODY_PRIMARY_CLASS = {"white", "black"}
BODY_SECONDARY_CLASS = {"brown"}
BODY_TERTIARY_CLASS = {"red", "yellow", "blue"}
BODY_RECESSIVE_CLASS = {"patchwork"}

def clean_color_alleles(color_alleles):
    cleaned = []

    for c in color_alleles:
        if c is None:
            continue

        c = str(c).lower()

        if c != "n/a":
            cleaned.append(c)

    return cleaned

def express_body_color_name(color_alleles):
    """
    Body color rule:

    white == black > brown > red == yellow == blue > patchwork

    Co-dominance:
    white + black = silver
    red + yellow = orange
    red + blue = purple
    yellow + blue = green

    Patchwork only appears if both alleles are patchwork.
    """
    alleles = clean_color_alleles(color_alleles)

    if len(alleles) == 0:
        return "n/a"

    # Handle two alleles explicitly.
    a = alleles[0]
    b = alleles[1] if len(alleles) > 1 else alleles[0]

    pair = {a, b}

    # Highest class: white/black.
    primary_present = [c for c in [a, b] if c in BODY_PRIMARY_CLASS]

    if len(primary_present) == 2:
        if pair == {"white", "black"}:
            return "silver"
        return primary_present[0]

    if len(primary_present) == 1:
        return primary_present[0]

    # Brown dominates anything below it.
    if "brown" in pair:
        return "brown"

    # Red/yellow/blue are co-dominant with each other.
    tertiary_present = [c for c in [a, b] if c in BODY_TERTIARY_CLASS]

    if len(tertiary_present) == 2:
        tertiary_pair = set(tertiary_present)

        if tertiary_pair == {"red", "yellow"}:
            return "orange"
        if tertiary_pair == {"red", "blue"}:
            return "purple"
        if tertiary_pair == {"yellow", "blue"}:
            return "green"

        # Same tertiary allele twice.
        return tertiary_present[0]

    if len(tertiary_present) == 1:
        return tertiary_present[0]

    # Patchwork is fully recessive.
    if a == "patchwork" and b == "patchwork":
        return "patchwork"

    return "n/a"

HAIR_COLOR_MAP = {
    "white":  (0.92, 0.90, 0.84),
    "black":  (0.05, 0.04, 0.04),
    "silver": (0.62, 0.62, 0.58),

    "brown":  (0.35, 0.19, 0.08),
    "blonde": (0.95, 0.76, 0.28),
    "red":    (0.72, 0.18, 0.10),
    "pink":   (0.95, 0.32, 0.62),
    "blue":   (0.18, 0.34, 0.80),

    "n/a":    (0.05, 0.04, 0.04),
}

# Smaller number = more dominant.
HAIR_DOMINANCE_RANK = {
    "white": 0,
    "black": 0,
    "brown": 1,
    "blonde": 2,
    "red": 3,
    "pink": 4,
    "blue": 5,
}

def express_hair_color_name(hair_color_alleles):
    """
    Hair color rule:

    white == black > brown > blonde > red > pink > blue

    Only co-dominance:
    white + black = silver

    Everything else is pure dominance/recessiveness.
    """
    alleles = clean_color_alleles(hair_color_alleles)

    if len(alleles) == 0:
        return "black"

    a = alleles[0]
    b = alleles[1] if len(alleles) > 1 else alleles[0]

    pair = {a, b}

    if pair == {"white", "black"}:
        return "silver"

    ranked = sorted(
        [a, b],
        key=lambda c: HAIR_DOMINANCE_RANK.get(c, 999)
    )

    return ranked[0]

def get_visual_phenotype(rock):
    """
    Clean drawing phenotype.
    Does not mutate rock.rock_cost.
    Does update rock.gender and rock.is_craisen for convenience.
    """
    v = {}

    # Gender
    gender_gene = str(rock.genes.get("gender", "00"))
    v["gender"] = "Male" if gender_gene == "01" else "Female"
    rock.gender = 1 if v["gender"] == "Male" else 0

    # Death status
    v["is_craisen"] = check_craisen(rock)
    rock.is_craisen = 1 if v["is_craisen"] else 0

    # Store allele names for every drawable gene.
    for gene_name in Rock_roll_dict:
        values = get_pair_values(rock, gene_name)
        allele_names = get_trait_allele_names(rock, gene_name)

        v[f"{gene_name}_values"] = values
        v[f"{gene_name}_alleles"] = allele_names

        if gene_name == "arms":
            normal_arm_pairs = values.count(1)
            muscle_arm_pairs = values.count(2)

            v["normal_arm_pairs"] = normal_arm_pairs
            v["muscle_arm_pairs"] = muscle_arm_pairs
            v["normal_arm_count"] = 2 * normal_arm_pairs
            v["muscle_arm_count"] = 2 * muscle_arm_pairs

            if normal_arm_pairs == 0 and muscle_arm_pairs == 0:
                v["arms"] = "n/a"
            else:
                pieces = []
                if normal_arm_pairs > 0:
                    pieces.append(f"{2 * normal_arm_pairs} arms")
                if muscle_arm_pairs > 0:
                    pieces.append(f"{2 * muscle_arm_pairs} muscle arms")
                v["arms"] = ", ".join(pieces)

        elif gene_name in ["eyes", "fuzz", "hair"]:
            trait_names = Rock_roll_dict[gene_name][2]
            expressed_idx = min(count_nonzero_alleles(rock, gene_name), len(trait_names) - 1)

            v[gene_name] = trait_names[expressed_idx]
            v[f"{gene_name}_count"] = expressed_idx

        else:
            v[gene_name] = get_primary_trait_name(rock, gene_name)

    # Gender interaction with facial hair.
    if v["gender"] == "Female" and v.get("facial_hair", "n/a") != "n/a":
        v["facial_hair"] = "peach fuzz"

    # Override body color and hair color with proper dominance/codominance rules.
    v["color"] = express_body_color_name(v.get("color_alleles", []))
    v["hair_color"] = express_hair_color_name(v.get("hair_color_alleles", []))

    return v

def ensure_rock_game_attributes(rock, imported=False, sold=False):
    """
    Adds gameplay attributes to a Rock object if they do not already exist.
    """
    if not hasattr(rock, "sold"):
        rock.sold = sold

    if not hasattr(rock, "imported"):
        rock.imported = imported

    if not hasattr(rock, "used_as_parent"):
        rock.used_as_parent = False

    if not hasattr(rock, "dead"):
        rock.dead = False

    if not hasattr(rock, "death_reason"):
        rock.death_reason = None

    if not hasattr(rock, "base_value"):
        rock.base_value = 0

    if not hasattr(rock, "sell_value"):
        rock.sell_value = 0

    if not hasattr(rock, "score_value"):
        rock.score_value = 0

    if not hasattr(rock, "is_craisen"):
        rock.is_craisen = 0

    if not hasattr(rock, "rock_cost"):
        rock.rock_cost = 0

    if not hasattr(rock, "gender"):
        rock.gender = None

    return rock

def evaluate_rock_value(rock):
    """
    Evaluate rock value for gameplay.

    Dead, craisen, sold, and bred-parent rocks are worth $0.
    """
    ensure_rock_game_attributes(rock)

    rock.rock_cost = 0
    rock.is_craisen = 0

    try:
        phenotype = get_rock_phenotype(rock)
    except Exception as e:
        phenotype = get_visual_phenotype(rock)
        rock.rock_cost = 1
        rock.is_craisen = 1 if phenotype.get("is_craisen", False) else 0
        print(f"Value warning for rock #{rock.id}: {e}")

    try:
        visual = get_visual_phenotype(rock)
        if visual.get("is_craisen", False):
            rock.is_craisen = 1
    except Exception:
        pass

    rock.base_value = max(0, int(getattr(rock, "rock_cost", 0)))

    if getattr(rock, "dead", False):
        rock.sell_value = 0
        rock.score_value = 0

    elif getattr(rock, "is_craisen", 0) == 1:
        rock.sell_value = 0
        rock.score_value = 0

    elif getattr(rock, "sold", False):
        rock.sell_value = 0
        rock.score_value = 0

    elif getattr(rock, "used_as_parent", False):
        rock.sell_value = 0
        rock.score_value = 0

    else:
        rock.sell_value = rock.base_value
        rock.score_value = rock.base_value

    return rock.sell_value


def rebuild_used_as_parent_flags(game):
    """
    Rebuild used_as_parent flags from existing child parent links.

    Any rock that appears as a parent of another rock gets used_as_parent=True.
    """
    for rock in game.rocks.values():
        ensure_rock_game_attributes(rock)
        rock.used_as_parent = False

    for child in game.rocks.values():
        if child.parents is not None:
            for parent_id in child.parents:
                if parent_id in game.rocks:
                    game.rocks[parent_id].used_as_parent = True

    return game

def evaluate_all_rocks(game):
    """
    Refresh all values.
    """
    rebuild_used_as_parent_flags(game)

    for rock in game.rocks.values():
        evaluate_rock_value(rock)

    return game

def ensure_rock_game_attributes(rock, imported=False, sold=False):
    """
    Adds gameplay attributes to a Rock object if they do not already exist.
    """
    if not hasattr(rock, "sold"):
        rock.sold = sold

    if not hasattr(rock, "imported"):
        rock.imported = imported

    if not hasattr(rock, "used_as_parent"):
        rock.used_as_parent = False

    if not hasattr(rock, "dead"):
        rock.dead = False

    if not hasattr(rock, "death_reason"):
        rock.death_reason = None

    if not hasattr(rock, "base_value"):
        rock.base_value = 0

    if not hasattr(rock, "sell_value"):
        rock.sell_value = 0

    if not hasattr(rock, "score_value"):
        rock.score_value = 0

    if not hasattr(rock, "is_craisen"):
        rock.is_craisen = 0

    if not hasattr(rock, "rock_cost"):
        rock.rock_cost = 0

    if not hasattr(rock, "gender"):
        rock.gender = None

    return rock

def express_gender_from_gene(gender_gene):
    """
    Gender expression rule.

    00 -> Female
    01 -> Male
    10 -> Male
    11 -> Male

    This makes allele order irrelevant.
    """
    gene = str(gender_gene)

    return 1 if "1" in gene else 0

def express_gender_name_from_gene(gender_gene):
    return "Male" if express_gender_from_gene(gender_gene) == 1 else "Female"

def get_rock_gender_value(rock):
    """
    Return gender as:
    1 = Male
    0 = Female

    Uses the actual gender gene so "10" and "01" both express male.
    """
    if rock is None:
        return 0

    gender_gene = str(rock.genes.get("gender", "00"))

    return express_gender_from_gene(gender_gene)

def get_rock_gender_name(rock):
    return "Male" if get_rock_gender_value(rock) == 1 else "Female"

def get_hair_color_from_alleles(hair_color_alleles):
    color_name = express_hair_color_name(hair_color_alleles)
    return HAIR_COLOR_MAP.get(color_name, HAIR_COLOR_MAP["white"])

def get_render_hair_color(ctx):
    return get_hair_color_from_alleles(
        ctx.v.get("hair_color_alleles", [ctx.v.get("hair_color", "black")])
    )

@dataclass
class RockRenderContext:
    ax: object
    rock: object
    v: dict
    rng: object
    py_rng: object
    body: object
    body_points: np.ndarray
    s: float
    body_color: object

    def __post_init__(self):
        self.xmin = float(np.min(self.body_points[:, 0]))
        self.xmax = float(np.max(self.body_points[:, 0]))
        self.ymin = float(np.min(self.body_points[:, 1]))
        self.ymax = float(np.max(self.body_points[:, 1]))

        self.width = self.xmax - self.xmin
        self.height = self.ymax - self.ymin

        self.cx = 0.5 * (self.xmin + self.xmax)
        self.cy = 0.5 * (self.ymin + self.ymax)

        self.unit = min(self.width, self.height)

    def xy(self, nx, ny):
        """
        Convert normalized body coordinates to actual plot coordinates.

        nx = 0 means body left edge
        nx = 1 means body right edge
        ny = 0 means body bottom
        ny = 1 means body top
        """
        x = self.xmin + nx * self.width
        y = self.ymin + ny * self.height
        return x, y
    
def make_body_points(shape_name, size_name, rng):
    """
    Generate the body outline points.
    """
    size_scale_map = {
        "medium": 1.00,
        "large": 1.60,
        "small": 0.70,
        "giant": 2.30,
        "missized": 1.30,
    }

    s = size_scale_map.get(size_name, 1.0)

    if shape_name == "triangle":
        base = np.array([
            [0.00, 1.05],
            [-1.05, -0.75],
            [1.05, -0.75],
        ]) * s

        points = []

        for i in range(3):
            a = base[i]
            b = base[(i + 1) % 3]

            for j in range(7):
                t = j / 7
                pt = (1 - t) * a + t * b
                #if size_name == "missized":
                #pt += rng.normal(0, 0.035 * s, size=2)
                points.append(pt)

        return np.array(points), s

    n_points = 34
    theta = np.linspace(0, 2 * np.pi, n_points, endpoint=False)

    if shape_name == "square":
        x = np.sign(np.cos(theta)) * np.abs(np.cos(theta)) ** 0.42
        y = np.sign(np.sin(theta)) * np.abs(np.sin(theta)) ** 0.42
    else:
        x = np.cos(theta)
        y = np.sin(theta)

    if shape_name == "circle":
        x_scale, y_scale = 1.00, 1.00
    elif shape_name == "oval":
        x_scale, y_scale = 0.82, 1.18
    elif shape_name == "oblong":
        x_scale, y_scale = 1.35, 0.72
    elif shape_name == "square":
        x_scale, y_scale = 1.00, 1.00
    else:
        x_scale, y_scale = 1.00, 1.00

    if size_name == "missized":
        x_scale *= 1.28
        y_scale *= 0.82

    wobble = 1 #
    if size_name == "missized":
      wobble += rng.normal(0, 0.055, n_points)

    points = np.column_stack([
        s * x_scale * x * wobble,
        s * y_scale * y * wobble
    ])

    return points, s

def get_body_color_from_alleles(color_alleles):
    color_name = express_body_color_name(color_alleles)
    return BODY_COLOR_MAP.get(color_name, BODY_COLOR_MAP["n/a"])

def get_wing_anchor_fraction(ctx):
    """
    Choose the vertical attachment height for wings.
    Sketch style: wings attach fairly high on the body.
    """
    shape = ctx.v.get("shape", "circle")

    if shape == "triangle":
        return 0.60
    elif shape == "oblong":
        return 0.66
    elif shape == "oval":
        return 0.66
    elif shape == "square":
        return 0.67
    else:
        return 0.66

def polygon_x_span_at_y(points, y):
    """
    Finds the left/right x-intersections of a horizontal line through a polygon.

    Returns (x_left, x_right).
    """
    xs = []
    n = len(points)

    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]

        # Skip horizontal edges to avoid duplicate weirdness.
        if abs(y2 - y1) < 1e-9:
            continue

        y_low = min(y1, y2)
        y_high = max(y1, y2)

        if y_low <= y <= y_high:
            t = (y - y1) / (y2 - y1)

            if 0 <= t <= 1:
                x = x1 + t * (x2 - x1)
                xs.append(x)

    if len(xs) < 2:
        # Fallback to bounding box.
        return float(np.min(points[:, 0])), float(np.max(points[:, 0]))

    xs = sorted(xs)
    return xs[0], xs[-1]

def body_span_at_fraction(ctx, y_frac):
    """
    Returns x-left, x-right, and y for a horizontal body slice.
    """
    y = ctx.ymin + y_frac * ctx.height
    x_left, x_right = polygon_x_span_at_y(ctx.body_points, y)

    return x_left, x_right, y

def clamp_inside_span(x, x_left, x_right, margin):
    return max(x_left + margin, min(x, x_right - margin))

def get_wing_layout(ctx):
    """
    Compute left/right wing anchors and scale information.
    Sketch style: rounded wings, slightly taller and moderately wide.
    """
    y_frac = get_wing_anchor_fraction(ctx)
    x_left, x_right, y = body_span_at_fraction(ctx, y_frac)

    local_span = x_right - x_left

    wing_w = max(0.52 * ctx.unit, 0.62 * local_span)
    wing_h = max(0.62 * ctx.unit, 0.78 * ctx.height)

    return {
        "left_anchor": (x_left, y),
        "right_anchor": (x_right, y),
        "span": local_span,
        "wing_w": wing_w,
        "wing_h": wing_h,
        "y_frac": y_frac
    }

def draw_single_wing(ctx, anchor, side=1, wing_w=1.0, wing_h=1.0):
    """
    Draw one rounded cartoon wing attached to the body edge.

    side:
    -1 = left wing
    +1 = right wing

    Style target:
    - rounded top arch
    - drooping outer wing
    - 3 little feather/finger tips
    - black outline, light fill
    """

    ax = ctx.ax
    x0, y0 = anchor

    sgn = side

    # Anchor points on the body edge
    root_top = (x0, y0 + 0.06 * wing_h)
    root_bot = (x0, y0 - 0.08 * wing_h)

    # Main structure points
    top_peak = (x0 + sgn * 0.30 * wing_w, y0 + 0.72 * wing_h)
    outer_top = (x0 + sgn * 0.82 * wing_w, y0 + 0.68 * wing_h)
    outer_mid = (x0 + sgn * 1.00 * wing_w, y0 + 0.10 * wing_h)
    lower_outer = (x0 + sgn * 0.88 * wing_w, y0 - 0.20 * wing_h)

    # Three feather/finger tips, like the sketch
    tip1 = (x0 + sgn * 0.92 * wing_w, y0 - 0.40 * wing_h)
    valley1 = (x0 + sgn * 0.76 * wing_w, y0 - 0.30 * wing_h)

    tip2 = (x0 + sgn * 0.72 * wing_w, y0 - 0.52 * wing_h)
    valley2 = (x0 + sgn * 0.58 * wing_w, y0 - 0.36 * wing_h)

    tip3 = (x0 + sgn * 0.50 * wing_w, y0 - 0.46 * wing_h)
    inner_return = (x0 + sgn * 0.28 * wing_w, y0 - 0.18 * wing_h)

    # Build a rounded wing outline using bezier curves + line segments
    verts = [
        root_top,  # start
        (x0 + sgn * 0.06 * wing_w, y0 + 0.24 * wing_h),   # control
        (x0 + sgn * 0.16 * wing_w, y0 + 0.48 * wing_h),   # control
        top_peak,                                          # top rise

        (x0 + sgn * 0.45 * wing_w, y0 + 0.86 * wing_h),   # control
        (x0 + sgn * 0.66 * wing_w, y0 + 0.80 * wing_h),   # control
        outer_top,                                         # top arch

        (x0 + sgn * 0.96 * wing_w, y0 + 0.46 * wing_h),   # control
        (x0 + sgn * 1.04 * wing_w, y0 + 0.24 * wing_h),   # control
        outer_mid,                                         # descend

        lower_outer,   # line down
        tip1,
        valley1,
        tip2,
        valley2,
        tip3,
        inner_return,

        (x0 + sgn * 0.10 * wing_w, y0 - 0.12 * wing_h),   # control back in
        root_bot,                                          # return near body
        root_top                                           # close
    ]

    codes = [
        Path.MOVETO,

        Path.CURVE4, Path.CURVE4, Path.CURVE4,
        Path.CURVE4, Path.CURVE4, Path.CURVE4,
        Path.CURVE4, Path.CURVE4, Path.CURVE4,

        Path.LINETO,
        Path.LINETO,
        Path.LINETO,
        Path.LINETO,
        Path.LINETO,
        Path.LINETO,
        Path.LINETO,

        Path.CURVE3,
        Path.CURVE3,
        Path.CLOSEPOLY
    ]

    wing_path = Path(verts, codes)

    wing_patch = PathPatch(
        wing_path,
        facecolor=(0.96, 0.96, 0.98, 0.95),
        edgecolor="black",
        linewidth=2.0,
        zorder=0,
        joinstyle="round",
        capstyle="round"
    )
    ax.add_patch(wing_patch)

    # A couple of inner feather support lines
    support_lines = [
        ((x0 + sgn * 0.08 * wing_w, y0 + 0.02 * wing_h), (x0 + sgn * 0.42 * wing_w, y0 + 0.56 * wing_h)),
        ((x0 + sgn * 0.12 * wing_w, y0 - 0.02 * wing_h), (x0 + sgn * 0.68 * wing_w, y0 + 0.10 * wing_h)),
        ((x0 + sgn * 0.16 * wing_w, y0 - 0.04 * wing_h), (x0 + sgn * 0.56 * wing_w, y0 - 0.26 * wing_h)),
    ]

    for p0, p1 in support_lines:
        ax.plot(
            [p0[0], p1[0]],
            [p0[1], p1[1]],
            color="black",
            linewidth=1.0,
            alpha=0.45,
            zorder=1
        )

    return wing_patch

def color_luminance(rgb):
    """
    Approximate perceived luminance for an RGB tuple in [0,1].
    """
    r, g, b = rgb
    return 0.2126 * r + 0.7152 * g + 0.0722 * b

def mix_colors(c1, c2, t=0.5):
    """
    Linear blend between two RGB colors.
    t=0 -> c1
    t=1 -> c2
    """
    c1 = np.array(c1, dtype=float)
    c2 = np.array(c2, dtype=float)
    return tuple(np.clip((1 - t) * c1 + t * c2, 0, 1))

def draw_wings(ctx):
    """
    Draw wings using ctx.

    Current rule:
    if v["wings"] != "n/a", draw one left wing and one right wing.
    """
    wing_trait = ctx.v.get("wings", "n/a")

    if wing_trait == "n/a":
        return None

    layout = get_wing_layout(ctx)

    left_anchor = layout["left_anchor"]
    right_anchor = layout["right_anchor"]
    wing_w = layout["wing_w"]
    wing_h = layout["wing_h"]

    left_wing = draw_single_wing(
        ctx,
        left_anchor,
        side=-1,
        wing_w=wing_w,
        wing_h=wing_h
    )

    right_wing = draw_single_wing(
        ctx,
        right_anchor,
        side=1,
        wing_w=wing_w,
        wing_h=wing_h
    )

    return {
        "left": left_wing,
        "right": right_wing,
        "layout": layout
    }

def get_fuzz_color(body_color):
    """
    Adaptive inner fuzz color:
    - lighter on dark rocks
    - darker on light rocks

    This sits on top of a black under-stroke for visibility.
    """
    lum = color_luminance(body_color)

    if lum < 0.45:
        return mix_colors(body_color, (1, 1, 1), t=0.55)
    else:
        return mix_colors(body_color, (0, 0, 0), t=0.45)

def polygon_perimeter(points):
    """
    Perimeter of a closed polygon.
    """
    pts = np.asarray(points)
    shifted = np.roll(pts, -1, axis=0)
    return np.sum(np.sqrt(np.sum((shifted - pts) ** 2, axis=1)))

def sample_polygon_boundary(points, n_samples, offset_frac=0.0):
    """
    Evenly sample points along a closed polygon boundary.
    Returns an array of shape (n_samples, 2).
    """
    pts = np.asarray(points)
    shifted = np.roll(pts, -1, axis=0)

    seg_vecs = shifted - pts
    seg_lens = np.sqrt(np.sum(seg_vecs ** 2, axis=1))
    cum = np.concatenate([[0], np.cumsum(seg_lens)])
    total = cum[-1]

    if total <= 1e-12:
        return np.repeat(pts[:1], n_samples, axis=0)

    samples = []
    start_dist = offset_frac * total

    for k in range(n_samples):
        d = (start_dist + k * total / n_samples) % total

        seg_idx = np.searchsorted(cum, d, side="right") - 1
        seg_idx = min(seg_idx, len(seg_lens) - 1)

        seg_start = cum[seg_idx]
        seg_len = seg_lens[seg_idx]

        if seg_len <= 1e-12:
            samples.append(pts[seg_idx].copy())
            continue

        t = (d - seg_start) / seg_len
        p = pts[seg_idx] + t * seg_vecs[seg_idx]
        samples.append(p)

    return np.array(samples)

def draw_fuzz(ctx):
    """
    Draw fuzz around the rock boundary.

    Expression rule:
    - 0 active fuzz alleles -> no fuzz
    - 1 active fuzz allele  -> small fuzz
    - 2 active fuzz alleles -> urchin-like spines

    Drawn behind the body.
    """
    fuzz_count = ctx.v.get("fuzz_count", 0)

    if fuzz_count <= 0:
        return None

    perimeter = polygon_perimeter(ctx.body_points)

    # Use a stable but shape-aware number of spikes.
    if fuzz_count == 1:
        # Small fuzz
        n_spikes = max(18, int(perimeter / (0.22 * ctx.unit)))
        n_spikes = min(n_spikes, 40)

        len_min = 0.05 * ctx.unit
        len_max = 0.12 * ctx.unit

        inner_lw = 1.0
        outer_lw = 1.8

        # Slight random wiggle
        angle_jitter = 0.28
        bent = False

    else:
        # Urchin mode
        n_spikes = max(10, int(perimeter / (0.34 * ctx.unit)))
        n_spikes = min(n_spikes, 24)

        len_min = 0.16 * ctx.unit
        len_max = 0.34 * ctx.unit

        inner_lw = 2.0
        outer_lw = 3.0

        angle_jitter = 0.18
        bent = True

    # Sample the outline evenly, with a deterministic offset.
    offset_frac = ((ctx.rock.id * 0.137) % 1.0)
    boundary_pts = sample_polygon_boundary(ctx.body_points, n_spikes, offset_frac=offset_frac)

    inner_color = get_fuzz_color(ctx.body_color)
    lines_drawn = []

    for p in boundary_pts:
        x0, y0 = p

        # Outward direction = from center to boundary point.
        dx = x0 - ctx.cx
        dy = y0 - ctx.cy
        norm = math.sqrt(dx * dx + dy * dy) + 1e-12
        ux = dx / norm
        uy = dy / norm

        # Add a small angular jitter.
        theta = math.atan2(uy, ux) + ctx.py_rng.uniform(-angle_jitter, angle_jitter)
        ux_j = math.cos(theta)
        uy_j = math.sin(theta)

        spike_len = ctx.py_rng.uniform(len_min, len_max)

        if not bent:
            # Small fuzz = single straight little hair
            x1 = x0 + ux_j * spike_len
            y1 = y0 + uy_j * spike_len

            # Black under-stroke
            line_bg, = ctx.ax.plot(
                [x0, x1],
                [y0, y1],
                color="black",
                linewidth=outer_lw,
                alpha=0.95,
                zorder=0.30,
                solid_capstyle="round"
            )

            # Adaptive inner stroke
            line_fg, = ctx.ax.plot(
                [x0, x1],
                [y0, y1],
                color=inner_color,
                linewidth=inner_lw,
                alpha=0.95,
                zorder=0.35,
                solid_capstyle="round"
            )

            lines_drawn.extend([line_bg, line_fg])

        else:
            # Urchin fuzz = stronger, slightly kinked spines
            mid_len = 0.58 * spike_len

            mx = x0 + ux_j * mid_len
            my = y0 + uy_j * mid_len

            # Small second bend
            theta2 = theta + ctx.py_rng.uniform(-0.20, 0.20)
            ux2 = math.cos(theta2)
            uy2 = math.sin(theta2)

            x2 = mx + ux2 * (spike_len - mid_len)
            y2 = my + uy2 * (spike_len - mid_len)

            line_bg, = ctx.ax.plot(
                [x0, mx, x2],
                [y0, my, y2],
                color="black",
                linewidth=outer_lw,
                alpha=0.98,
                zorder=0.30,
                solid_capstyle="round"
            )

            line_fg, = ctx.ax.plot(
                [x0, mx, x2],
                [y0, my, y2],
                color=inner_color,
                linewidth=inner_lw,
                alpha=0.98,
                zorder=0.35,
                solid_capstyle="round"
            )

            lines_drawn.extend([line_bg, line_fg])

    return lines_drawn

def get_halo_layout(ctx):
    """
    Compute halo placement above the rock.

    Goal:
    - centered above the head
    - about 40% of the rock height above the top
    - scaled to body width/height
    """
    halo_type = ctx.v.get("halos", "n/a")

    if halo_type == "n/a":
        return None

    cx = ctx.cx
    top_y = ctx.ymax

    # Place halo center about 40% of rock height above the top.
    cy = top_y + 0.40 * ctx.height

    # Halo size relative to rock.
    halo_w = 0.62 * ctx.width
    halo_h = 0.14 * ctx.height

    return {
        "type": halo_type,
        "center": (cx, cy),
        "width": halo_w,
        "height": halo_h,
        "top_y": top_y
    }

def draw_halo(ctx):
    """
    Draw a single golden halo with black outline.

    Style:
    - black outer outline for clarity
    - gold inner outline
    - simple ellipse
    """
    halo_type = ctx.v.get("halos", "n/a")

    if halo_type == "n/a":
        return None

    layout = get_halo_layout(ctx)

    cx, cy = layout["center"]
    hw = layout["width"]
    hh = layout["height"]

    # Outer black outline
    halo_black = Ellipse(
        (cx, cy),
        width=hw,
        height=hh,
        facecolor="none",
        edgecolor="black",
        linewidth=3.2,
        zorder=12
    )
    ctx.ax.add_patch(halo_black)

    # Inner gold outline
    halo_gold = Ellipse(
        (cx, cy),
        width=0.94 * hw,
        height=0.82 * hh,
        facecolor="none",
        edgecolor="gold",
        linewidth=2.2,
        zorder=13
    )
    ctx.ax.add_patch(halo_gold)

    return layout

def get_ion_stone_layout(ctx):
    """
    Compute an orbit path above the rock head for an ion stone.
    """
    stone_trait = ctx.v.get("stones", "n/a")

    if stone_trait == "n/a":
        return None

    # Orbit sits above the top half of the rock.
    orbit_cx = ctx.cx
    orbit_cy = ctx.ymax + 0.20 * ctx.height

    orbit_w = 0.95 * ctx.width
    orbit_h = 0.42 * ctx.height

    # Arc span over the head.
    theta1 = 20
    theta2 = 160

    # Choose a deterministic stone position along the arc,
    # biased a bit to the upper-left like your sketch.
    stone_angle_deg = ctx.py_rng.uniform(130, 155)
    t = math.radians(stone_angle_deg)

    sx = orbit_cx + 0.5 * orbit_w * math.cos(t)
    sy = orbit_cy + 0.5 * orbit_h * math.sin(t)

    # Stone size varies a little, but not wildly.
    stone_size = ctx.py_rng.uniform(0.10, 0.15) * ctx.unit

    return {
        "orbit_center": (orbit_cx, orbit_cy),
        "orbit_w": orbit_w,
        "orbit_h": orbit_h,
        "theta1": theta1,
        "theta2": theta2,
        "stone_center": (sx, sy),
        "stone_size": stone_size,
    }

def get_ion_stone_color(ctx):
    """
    Pick a pleasant little gem color.
    Deterministic per rock through ctx.py_rng.
    """
    palette = [
        (0.74, 0.88, 1.00),  # pale blue
        (0.90, 0.78, 1.00),  # lavender
        (1.00, 0.82, 0.84),  # rose
        (0.82, 1.00, 0.86),  # mint
        (1.00, 0.90, 0.66),  # amber
        (0.92, 0.92, 1.00),  # pearl
    ]

    idx = ctx.py_rng.randint(0, len(palette) - 1)
    return palette[idx]

def make_ion_stone_polygon(center, size, rng, n_sides=None):
    """
    Make a small irregular gemstone polygon.
    """
    cx, cy = center

    if n_sides is None:
        n_sides = rng.randint(4, 6)

    theta0 = rng.uniform(0, 2 * math.pi)
    angles = np.linspace(0, 2 * math.pi, n_sides, endpoint=False) + theta0

    points = []
    for a in angles:
        r = size * rng.uniform(0.80, 1.18)
        points.append([
            cx + r * math.cos(a),
            cy + r * math.sin(a)
        ])

    return points

def draw_stones(ctx):
    """
    Draw an ion stone trait:
    - one small floating stone
    - one orbit arc over the head
    - slight variation in shape and size
    """
    stone_trait = ctx.v.get("stones", "n/a")

    if stone_trait == "n/a":
        return None

    layout = get_ion_stone_layout(ctx)

    orbit_cx, orbit_cy = layout["orbit_center"]
    orbit_w = layout["orbit_w"]
    orbit_h = layout["orbit_h"]
    theta1 = layout["theta1"]
    theta2 = layout["theta2"]
    stone_center = layout["stone_center"]
    stone_size = layout["stone_size"]

    # Orbit arc
    orbit_arc = Arc(
        (orbit_cx, orbit_cy),
        width=orbit_w,
        height=orbit_h,
        theta1=theta1,
        theta2=theta2,
        color="black",
        linewidth=1.8,
        zorder=14
    )
    ctx.ax.add_patch(orbit_arc)

    # Ion stone shape
    gem_points = make_ion_stone_polygon(
        stone_center,
        stone_size,
        ctx.py_rng
    )

    gem_color = get_ion_stone_color(ctx)

    gem = Polygon(
        gem_points,
        closed=True,
        facecolor=gem_color,
        edgecolor="black",
        linewidth=1.4,
        zorder=15,
        joinstyle="round"
    )
    ctx.ax.add_patch(gem)

    # Tiny highlight for gem shine
    gx, gy = stone_center
    highlight = Circle(
        (gx - 0.20 * stone_size, gy + 0.20 * stone_size),
        radius=0.20 * stone_size,
        facecolor="white",
        edgecolor="none",
        alpha=0.45,
        zorder=16
    )
    ctx.ax.add_patch(highlight)

    return {
        "layout": layout,
        "orbit_arc": orbit_arc,
        "gem": gem
    }

def get_tail_layout(ctx):
    """
    Compute a tail anchor on the lower body edge.

    Tail comes from the bottom / lower-side region,
    with slight deterministic side variation.
    """
    tail_trait = ctx.v.get("tails", "n/a")

    if tail_trait == "n/a":
        return None

    shape = ctx.v.get("shape", "circle")

    # Use a low body slice.
    if shape == "triangle":
        y_frac = 0.14
    elif shape == "oblong":
        y_frac = 0.18
    else:
        y_frac = 0.16

    x_left, x_right, y = body_span_at_fraction(ctx, y_frac)
    span = x_right - x_left

    # Choose side deterministically but with variety.
    side = -1 if (ctx.rock.id % 2 == 0) else 1

    # Anchor can be centered-ish or offset toward a side.
    mode = ctx.rock.id % 3

    if mode == 0:
        anchor_x = ctx.cx
    elif mode == 1:
        anchor_x = ctx.cx + side * 0.18 * span
    else:
        anchor_x = ctx.cx + side * 0.28 * span

    anchor_x = max(x_left + 0.06 * span, min(x_right - 0.06 * span, anchor_x))

    # Tail scale
    tail_len = ctx.py_rng.uniform(0.42, 0.65) * ctx.unit
    tail_drop = ctx.py_rng.uniform(0.30, 0.48) * ctx.unit
    tip_size = ctx.py_rng.uniform(0.07, 0.11) * ctx.unit

    return {
        "anchor": (anchor_x, y),
        "side": side,
        "tail_len": tail_len,
        "tail_drop": tail_drop,
        "tip_size": tip_size,
        "y_frac": y_frac,
        "span": span,
    }

def make_tail_tip(center, size, side=1, rng=None):
    """
    Small tail tip shape.
    Slight variation between pebble/diamond-ish shapes.
    """
    if rng is None:
        rng = random.Random(0)

    cx, cy = center
    mode = rng.randint(0, 2)

    if mode == 0:
        # Little rounded pebble-like diamond
        pts = [
            [cx, cy + 1.00 * size],
            [cx + 0.85 * size, cy + 0.35 * size],
            [cx + 0.70 * size, cy - 0.75 * size],
            [cx - 0.55 * size, cy - 0.70 * size],
            [cx - 0.90 * size, cy + 0.10 * size],
        ]
    elif mode == 1:
        # Squatter polygon
        pts = [
            [cx - 0.75 * size, cy + 0.20 * size],
            [cx - 0.20 * size, cy + 0.95 * size],
            [cx + 0.80 * size, cy + 0.30 * size],
            [cx + 0.70 * size, cy - 0.65 * size],
            [cx - 0.35 * size, cy - 0.80 * size],
        ]
    else:
        # Simple diamond-ish point
        pts = [
            [cx, cy + 1.00 * size],
            [cx + 0.85 * size, cy],
            [cx, cy - 0.95 * size],
            [cx - 0.80 * size, cy],
        ]

    return pts

def draw_tail(ctx):
    """
    Draw a curved tail attached to the lower body.

    Style:
    - starts from bottom/lower-side of rock
    - curves outward and down
    - ends in a small tip shape
    - slight deterministic variation
    """
    tail_trait = ctx.v.get("tails", "n/a")

    if tail_trait == "n/a":
        return None

    layout = get_tail_layout(ctx)

    x0, y0 = layout["anchor"]
    side = layout["side"]
    tail_len = layout["tail_len"]
    tail_drop = layout["tail_drop"]
    tip_size = layout["tip_size"]

    # Control points for a nice curving tail.
    c1 = (
        x0 + side * 0.02 * ctx.unit,
        y0 - 0.18 * tail_drop
    )

    c2 = (
        x0 + side * 0.10 * tail_len,
        y0 - 0.68 * tail_drop
    )

    end = (
        x0 + side * tail_len,
        y0 - tail_drop
    )

    # Optional extra bend / curl
    bend = ctx.rock.id % 3

    if bend == 0:
        c2 = (c2[0] + side * 0.10 * tail_len, c2[1] - 0.05 * ctx.unit)
    elif bend == 1:
        c2 = (c2[0] - side * 0.06 * tail_len, c2[1] + 0.04 * ctx.unit)
    else:
        c1 = (c1[0] + side * 0.05 * tail_len, c1[1])

    tail_path = Path(
        [ (x0, y0), c1, c2, end ],
        [ Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4 ]
    )

    tail = PathPatch(
        tail_path,
        facecolor="none",
        edgecolor="black",
        linewidth=2.6,
        zorder=0.6,
        capstyle="round",
        joinstyle="round"
    )
    ctx.ax.add_patch(tail)

    # Tail tip
    tip_center = end
    tip_pts = make_tail_tip(tip_center, tip_size, side=side, rng=ctx.py_rng)

    tip = Polygon(
        tip_pts,
        closed=True,
        facecolor=ctx.body_color,
        edgecolor="black",
        linewidth=1.4,
        zorder=0.7,
        joinstyle="round"
    )
    ctx.ax.add_patch(tip)

    return {
        "layout": layout,
        "tail": tail,
        "tip": tip,
        "end": end
    }

def get_horn_layout(ctx):
    """
    Compute consistent horn placement from the head shape.

    Horns are placed near the top of the body using the width of the rock
    at a high body slice. Only placement and scale depend on the rock;
    horn shape itself stays canned/consistent.
    """
    horn_type = ctx.v.get("horns", "n/a")

    if horn_type == "n/a":
        return None

    shape = ctx.v.get("shape", "circle")

    # Use a high slice to determine "head width"
    if shape == "triangle":
        y_frac = 0.80
    elif shape == "oblong":
        y_frac = 0.84
    else:
        y_frac = 0.83

    x_left, x_right, y_band = body_span_at_fraction(ctx, y_frac)
    local_width = x_right - x_left

    top_y = ctx.ymax

    # Horn anchors sit a bit inward from the high-side body edges.
    inset = 0.16 * local_width
    left_base = (x_left + inset, y_band + 0.02 * ctx.unit)
    right_base = (x_right - inset, y_band + 0.02 * ctx.unit)

    # Scale from rock size, but clamp so they stay visually consistent.
    horn_scale = np.clip(0.95 * ctx.unit, 0.75, 1.35)

    return {
        "type": horn_type,
        "left_base": left_base,
        "right_base": right_base,
        "top_y": top_y,
        "local_width": local_width,
        "scale": horn_scale
    }

def make_canned_horn_template():
    """
    Canonical horn shape in local coordinates.

    Base is centered near (0, 0).
    Horn points upward with a slight outward lean.
    This template is used for both horns; right horn is mirrored.
    """
    pts = np.array([
        [-0.18,  0.00],   # base left
        [-0.10,  0.18],
        [-0.04,  0.42],
        [ 0.02,  0.74],   # tip
        [ 0.12,  0.48],
        [ 0.16,  0.20],
        [ 0.18,  0.00],   # base right
        [ 0.08,  0.10],   # inner ridge start
        [ 0.00,  0.28],   # inner ridge mid
    ])
    return pts

def draw_single_horn(ctx, base, side=1, horn_scale=1.0):
    """
    Draw one horn using a canned horn template.

    side:
    -1 = left horn
    +1 = right horn

    Only placement and scale vary. Shape stays consistent.
    """
    bx, by = base
    template = make_canned_horn_template().copy()

    # Scale horn to body size.
    w = 0.42 * horn_scale
    h = 0.62 * horn_scale

    # Transform template into horn coordinates.
    horn_pts = template.copy()
    horn_pts[:, 0] *= w
    horn_pts[:, 1] *= h

    # Mirror for left horn so both point outward.
    if side == -1:
        horn_pts[:, 0] *= -1

    # Small outward lean shift.
    horn_pts[:, 0] += side * 0.03 * horn_scale

    # Translate to base point.
    horn_pts[:, 0] += bx
    horn_pts[:, 1] += by

    # Outer horn polygon uses first 7 points.
    poly_pts = horn_pts[:7]

    horn = Polygon(
        poly_pts,
        closed=True,
        facecolor="tan",
        edgecolor="black",
        linewidth=1.3,
        zorder=12,
        joinstyle="round"
    )
    ctx.ax.add_patch(horn)

    # Inner ridge line from remaining points.
    ridge = horn_pts[7:]
    ctx.ax.plot(
        ridge[:, 0],
        ridge[:, 1],
        color="black",
        linewidth=0.8,
        alpha=0.35,
        zorder=13
    )

    return horn

def draw_horns(ctx):
    """
    Draw classic two horns using a canned horn shape and head-based placement.
    """
    horn_type = ctx.v.get("horns", "n/a")

    if horn_type == "n/a":
        return None

    layout = get_horn_layout(ctx)

    left_horn = draw_single_horn(
        ctx,
        layout["left_base"],
        side=-1,
        horn_scale=layout["scale"]
    )

    right_horn = draw_single_horn(
        ctx,
        layout["right_base"],
        side=1,
        horn_scale=layout["scale"]
    )

    return {
        "type": horn_type,
        "left": left_horn,
        "right": right_horn,
        "layout": layout
    }

def draw_patchwork(ax, body_patch, color_alleles, s, rng):
    """
    Draw dispersed random patches only if body color expresses as patchwork.

    Since patchwork is recessive, this should only happen for:
    patchwork + patchwork.
    """
    body_color_name = express_body_color_name(color_alleles)

    if body_color_name != "patchwork":
        return

    patch_palette = [
        BODY_COLOR_MAP["white"],
        BODY_COLOR_MAP["black"],
        BODY_COLOR_MAP["silver"],
        BODY_COLOR_MAP["brown"],
        BODY_COLOR_MAP["red"],
        BODY_COLOR_MAP["yellow"],
        BODY_COLOR_MAP["blue"],
        BODY_COLOR_MAP["orange"],
        BODY_COLOR_MAP["green"],
        BODY_COLOR_MAP["purple"],
    ]

    n_patches = random.randint(1,5) * random.randint(1,4) + random.randint(1,6)

    for i in range(n_patches):
        cx = rng.uniform(-0.85 * s, 0.85 * s)
        cy = rng.uniform(-0.75 * s, 0.85 * s)

        r = rng.uniform(0.16 * s, 0.40 * s)
        n_sides = int(rng.integers(5, 9))

        theta0 = rng.uniform(0, 2 * np.pi)
        angles = np.linspace(0, 2 * np.pi, n_sides, endpoint=False) + theta0

        points = []

        for a in angles:
            rr = r * rng.uniform(0.55, 1.15)
            points.append([
                cx + rr * np.cos(a),
                cy + rr * np.sin(a)
            ])

        patch = Polygon(
            points,
            closed=True,
            facecolor=patch_palette[i % len(patch_palette)],
            edgecolor="black",
            linewidth=0.35,
            alpha=0.55,
            zorder=2.5
        )

        patch.set_clip_path(body_patch)
        ax.add_patch(patch)

def get_hair_layout(ctx):
    """
    Compute head-aware anchor geometry for hair.
    """

    hair_type = ctx.v.get("hair", "n/a")

    if hair_type == "n/a":
        return None

    # Top cap width
    top_left, top_right, top_y = body_span_at_fraction(ctx, 0.88)

    # Forehead band for front hairline
    head_left, head_right, head_y = body_span_at_fraction(ctx, 0.74)

    top_span = top_right - top_left
    head_span = head_right - head_left

    return {
        "type": hair_type,
        "top_left": top_left,
        "top_right": top_right,
        "top_y": top_y,
        "head_left": head_left,
        "head_right": head_right,
        "head_y": head_y,
        "top_span": top_span,
        "head_span": head_span,
        "cx": ctx.cx,
        "cy": ctx.cy
    }

def get_render_hair_color(ctx):
    return get_hair_color_from_alleles(
        ctx.v.get("hair_color_alleles", [ctx.v.get("hair_color", "black")])
    )

def draw_curly_overlay_in_box(
    ctx,
    x_min,
    x_max,
    y_min,
    y_max,
    hair_color,
    n_curls=10,
    curl_scale=0.10,
    zorder=60,
    salt="head_curls"
):
    """
    Draw randomized semicircle curl marks inside an approximate hair region.

    The region is a simple bounding box, but the arcs are decorative and
    read well over both head hair and facial hair.
    """
    if x_max <= x_min or y_max <= y_min:
        return []

    rng = deterministic_rng_for_rock(ctx.rock, salt=salt)

    light = adjust_color_brightness(hair_color, 1.45)
    dark = adjust_color_brightness(hair_color, 0.65)

    curls = []

    box_w = x_max - x_min
    box_h = y_max - y_min

    for i in range(n_curls):
        x = rng.uniform(x_min + 0.08 * box_w, x_max - 0.08 * box_w)
        y = rng.uniform(y_min + 0.15 * box_h, y_max - 0.10 * box_h)

        size = rng.uniform(0.65, 1.15) * curl_scale * ctx.unit

        # Alternate light/dark so curls show on many colors.
        color = light if i % 2 == 0 else dark

        # Mostly semicircles, slightly rotated.
        angle = rng.uniform(-25, 25)

        # Randomize arc direction a bit.
        if rng.random() < 0.5:
            theta1, theta2 = 0, 180
        else:
            theta1, theta2 = 180, 360

        arc = draw_curl_arc(
            ctx.ax,
            x=x,
            y=y,
            w=size,
            h=0.72 * size,
            color=color,
            angle=angle,
            theta1=theta1,
            theta2=theta2,
            linewidth=max(0.8, 1.05 * ctx.unit),
            alpha=0.9,
            zorder=zorder
        )

        curls.append(arc)

    return curls

def draw_head_hair_curls(ctx, hair_color, layout, hair_type):
    """
    Add curl marks over head hair if hair_texture is curly.
    """
    if not rock_texture_is_curly(ctx):
        return []

    head_left = layout["head_left"]
    head_right = layout["head_right"]
    head_y = layout["head_y"]
    top_y = layout["top_y"]
    cx = layout["cx"]
    head_span = layout["head_span"]

    # Approximate curl region depending on hairstyle.
    if hair_type == "hair":
        x_min = head_left + 0.03 * head_span
        x_max = head_right - 0.03 * head_span
        y_min = head_y - 0.10 * ctx.unit
        y_max = top_y + 0.25 * ctx.unit
        n_curls = 8

    elif hair_type == "double hair":
        x_min = head_left - 0.20 * head_span
        x_max = head_right + 0.20 * head_span
        y_min = head_y - 0.18 * ctx.unit
        y_max = top_y + 0.30 * ctx.unit
        n_curls = 13

    else:
        x_min = cx - 0.40 * head_span
        x_max = cx + 0.40 * head_span
        y_min = head_y - 0.10 * ctx.unit
        y_max = top_y + 0.25 * ctx.unit
        n_curls = 8

    return draw_curly_overlay_in_box(
        ctx,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        hair_color=hair_color,
        n_curls=n_curls,
        curl_scale=0.095,
        zorder=72,
        salt=f"head_curls_{hair_type}"
    )

def draw_hair(ctx, rock, v):
    """
    Draw canned hairstyles inspired by the sketch.

    Styles:
    - femme_side
    - femme_long
    - masc_short
    - masc_spike
    """
    hair_type = ctx.v.get("hair", "n/a")
    #print(hair_type)
    #gender = rock.genes["gender"]
    gender = v["gender"]
    #print(gender)

    if hair_type == "n/a":
        return None

    layout = get_hair_layout(ctx)
    hair_color = get_render_hair_color(ctx)

    top_left = layout["top_left"]
    top_right = layout["top_right"]
    top_y = layout["top_y"]

    head_left = layout["head_left"]
    head_right = layout["head_right"]
    head_y = layout["head_y"]

    cx = layout["cx"]
    head_span = layout["head_span"]
    top_span = layout["top_span"]

    z = 11

    pieces = []

    def add_arc_cap(
        left_x,
        right_x,
        base_y,
        top_bump=0.12,
        lower_dip=0.08,
        zorder=11
    ):
        """
        Shared hair cap.

        It uses:
        - a top arc rising over the head
        - a lower arc dipping across the forehead

        This makes the visible top hairline smooth from left to right.
        """

        top_peak_y = top_y + top_bump * ctx.unit
        lower_y = base_y - lower_dip * ctx.unit

        verts = [
            # Start at left hairline
            (left_x, base_y),

            # Top arc: left -> right
            (left_x + 0.20 * (right_x - left_x), top_peak_y),
            (cx - 0.15 * (right_x - left_x), top_peak_y),
            (cx, top_peak_y),

            (cx + 0.15 * (right_x - left_x), top_peak_y),
            (right_x - 0.20 * (right_x - left_x), top_peak_y),
            (right_x, base_y),

            # Lower forehead arc: right -> left
            (right_x - 0.18 * (right_x - left_x), lower_y),
            (cx + 0.18 * (right_x - left_x), lower_y),
            (cx, lower_y),

            (cx - 0.18 * (right_x - left_x), lower_y),
            (left_x + 0.18 * (right_x - left_x), lower_y),
            (left_x, base_y),
        ]

        codes = [
            Path.MOVETO,

            Path.CURVE4,
            Path.CURVE4,
            Path.CURVE4,

            Path.CURVE4,
            Path.CURVE4,
            Path.CURVE4,

            Path.CURVE4,
            Path.CURVE4,
            Path.CURVE4,

            Path.CURVE4,
            Path.CURVE4,
            Path.CURVE4,
        ]

        cap = PathPatch(
            Path(verts, codes),
            facecolor=hair_color,
            edgecolor="black",
            linewidth=1.4,
            zorder=zorder,
            joinstyle="round",
            capstyle="round"
        )

        ctx.ax.add_patch(cap)
        return cap

    # Shared cap geometry.
    cap_left = head_left + 0.03 * head_span
    cap_right = head_right - 0.03 * head_span
    cap_base_y = head_y + 0.04 * ctx.unit

    # --------------------------------------------------
    # 1) FEMME SIDE — side sweep with one long lock
    # --------------------------------------------------
    if hair_type == "hair" and gender == "Female":
        cap = add_arc_cap(
            cap_left,
            cap_right,
            cap_base_y,
            zorder=z
        )
        pieces.append(cap)

        # Long side lock to the right
        lock_pts = [
            [cx + 0.08 * head_span, top_y + 0.16 * ctx.unit],
            [cx + 0.34 * head_span, top_y + 0.26 * ctx.unit],
            [cx + 0.74 * head_span, head_y + 0.12 * ctx.unit],
            [cx + 0.86 * head_span, head_y - 0.08 * ctx.unit],
            [cx + 0.62 * head_span, head_y - 0.04 * ctx.unit],
            [cx + 0.46 * head_span, head_y + 0.10 * ctx.unit],
            [cx + 0.28 * head_span, head_y + 0.18 * ctx.unit],
        ]

        lock = Polygon(
            lock_pts,
            closed=True,
            facecolor=hair_color,
            edgecolor="black",
            linewidth=1.4,
            zorder=z + 1,
            joinstyle="round"
        )
        ctx.ax.add_patch(lock)
        pieces.append(lock)

        pieces.extend(
            draw_head_hair_curls(
                ctx,
                hair_color=hair_color,
                layout=layout,
                hair_type=hair_type
            )
        )

        return pieces

    # --------------------------------------------------
    # 2) FEMME LONG — twin long draping locks
    # --------------------------------------------------
    elif hair_type == "double hair" and gender == "Female":
        cap = add_arc_cap(
            cap_left,
            cap_right,
            cap_base_y,
            zorder=z
        )
        pieces.append(cap)

        left_lock_pts = [
            [head_left + 0.18 * head_span, top_y + 0.14 * ctx.unit],
            [head_left - 0.14 * head_span, top_y + 0.24 * ctx.unit],
            [head_left - 0.40 * head_span, head_y + 0.06 * ctx.unit],
            [head_left - 0.54 * head_span, head_y - 0.16 * ctx.unit],
            [head_left - 0.26 * head_span, head_y - 0.14 * ctx.unit],
            [head_left - 0.08 * head_span, head_y + 0.02 * ctx.unit],
            [head_left + 0.12 * head_span, head_y + 0.14 * ctx.unit],
        ]

        right_lock_pts = [
            [head_right - 0.18 * head_span, top_y + 0.14 * ctx.unit],
            [head_right + 0.14 * head_span, top_y + 0.24 * ctx.unit],
            [head_right + 0.40 * head_span, head_y + 0.06 * ctx.unit],
            [head_right + 0.54 * head_span, head_y - 0.16 * ctx.unit],
            [head_right + 0.26 * head_span, head_y - 0.14 * ctx.unit],
            [head_right + 0.08 * head_span, head_y + 0.02 * ctx.unit],
            [head_right - 0.12 * head_span, head_y + 0.14 * ctx.unit],
        ]

        left_lock = Polygon(
            left_lock_pts,
            closed=True,
            facecolor=hair_color,
            edgecolor="black",
            linewidth=1.4,
            zorder=z + 1,
            joinstyle="round"
        )

        right_lock = Polygon(
            right_lock_pts,
            closed=True,
            facecolor=hair_color,
            edgecolor="black",
            linewidth=1.4,
            zorder=z + 1,
            joinstyle="round"
        )

        ctx.ax.add_patch(left_lock)
        ctx.ax.add_patch(right_lock)
        pieces.extend([left_lock, right_lock])

        pieces.extend(
            draw_head_hair_curls(
                ctx,
                hair_color=hair_color,
                layout=layout,
                hair_type=hair_type
            )
        )

        return pieces

    # --------------------------------------------------
    # 3) MASC SHORT — short top hair
    # --------------------------------------------------
    elif hair_type == "hair" and gender == "Male":
        cap = add_arc_cap(
            cap_left,
            cap_right,
            cap_base_y,
            zorder=z
        )
        pieces.append(cap)

        pieces.extend(
            draw_head_hair_curls(
                ctx,
                hair_color=hair_color,
                layout=layout,
                hair_type=hair_type
            )
        )

        return pieces

    # --------------------------------------------------
    # 4) MASC SPIKE — fuller cap with side spike
    # --------------------------------------------------
    elif hair_type == "double hair" and gender == "Male":
        cap = add_arc_cap(
            cap_left,
            cap_right,
            cap_base_y,
            zorder=z
          )
        pieces.append(cap)

        spike_pts = [
            [cx + 0.08 * head_span, top_y + 0.10 * ctx.unit],
            [cx + 0.25 * head_span, top_y + 0.32 * ctx.unit],
            [head_right + 0.16 * head_span, top_y + 0.42 * ctx.unit],
            [head_right + 0.06 * head_span, head_y + 0.12 * ctx.unit],
            [cx + 0.18 * head_span, head_y + 0.02 * ctx.unit],
        ]

        spike = Polygon(
            spike_pts,
            closed=True,
            facecolor=hair_color,
            edgecolor="black",
            linewidth=1.4,
            zorder=z + 1,
            joinstyle="round"
        )
        ctx.ax.add_patch(spike)
        pieces.append(spike)

        pieces.extend(
            draw_head_hair_curls(
                ctx,
                hair_color=hair_color,
                layout=layout,
                hair_type=hair_type
            )
        )

        return pieces

    return pieces

def get_ear_layout(ctx):
    """
    Compute ear anchors and scale from the rock head.

    Ears use canned shapes, while placement/scale comes from the body.
    """
    ear_type = ctx.v.get("ears", "n/a")

    if ear_type == "n/a":
        return None

    shape = ctx.v.get("shape", "circle")

    # Different ear types want different vertical bands.
    if ear_type in ["antannae", "antanna"]:
        y_frac = 0.82 if shape != "triangle" else 0.76
    elif ear_type in ["ears", "ear"]:
        y_frac = 0.66 if shape != "triangle" else 0.58
    elif ear_type in ["ogre", "ogres"]:
        y_frac = 0.80 if shape != "triangle" else 0.74
    elif ear_type in ["goblins", "goblin"]:
        y_frac = 0.82 if shape != "triangle" else 0.76
    else:
        y_frac = 0.75

    x_left, x_right, y = body_span_at_fraction(ctx, y_frac)
    local_width = x_right - x_left

    # Inset slightly from the exact edge for top-attached ear types.
    if ear_type in ["antannae", "antanna", "ogre", "ogres", "goblins", "goblin"]:
        inset = 0.10 * local_width
        left_base = (x_left + inset, y + 0.01 * ctx.unit)
        right_base = (x_right - inset, y + 0.01 * ctx.unit)
    else:
        # Rounded side ears sit right at the side region.
        left_base = (x_left, y)
        right_base = (x_right, y)

    # Moderate size, not too wild.
    if ear_type in ["ears", "ear"]:
        scale = np.clip(0.85 * ctx.unit, 0.70, 1.25)
    elif ear_type in ["ogre", "ogres"]:
        scale = np.clip(0.95 * ctx.unit, 0.75, 1.35)
    elif ear_type in ["goblins", "goblin"]:
        scale = np.clip(1.00 * ctx.unit, 0.80, 1.40)
    else:
        scale = np.clip(0.95 * ctx.unit, 0.75, 1.35)

    return {
        "type": ear_type,
        "left_base": left_base,
        "right_base": right_base,
        "y_frac": y_frac,
        "local_width": local_width,
        "scale": scale
    }

def transform_template_points(points, base, side=1, sx=1.0, sy=1.0, dx=0.0, dy=0.0):
    """
    Transform a canned local template:
    - scale x/y
    - mirror in x for left/right
    - translate to base
    """
    pts = np.array(points, dtype=float).copy()

    pts[:, 0] *= sx
    pts[:, 1] *= sy

    if side == -1:
        pts[:, 0] *= -1

    pts[:, 0] += dx + base[0]
    pts[:, 1] += dy + base[1]

    return pts

def make_round_ear_template():
    """
    Human-esque ear.
    Rounded with a slightly narrower lower attachment and fuller upper body.
    Local coordinates.
    """
    return np.array([
        [ 0.00, -0.10],   # lower attach
        [ 0.10, -0.02],
        [ 0.18,  0.10],
        [ 0.20,  0.26],
        [ 0.14,  0.40],
        [ 0.02,  0.48],   # top
        [-0.08,  0.42],
        [-0.14,  0.26],
        [-0.12,  0.10],
        [-0.06, -0.02],
    ])

def make_ogre_ear_template():
    """
    Shrek-like ogre ear.
    Broad, flared outward, with a trumpet/tube-like silhouette.
    """
    return np.array([
        [ 0.00, -0.06],   # attach point
        [ 0.10,  0.00],
        [ 0.28,  0.08],
        [ 0.46,  0.08],
        [ 0.60,  -0.18],   # outward flare
        [ 0.73,  0.08],
        [ 0.78,  0.35],
        [ 0.60,  0.58],
        [ 0.48,  0.38],
        [ 0.28,  0.34],
        [ 0.12,  0.24],
        [ 0.02,  0.12],
        [-0.02,  0.02],
    ])

def make_goblin_ear_template():
    """
    Elfish goblin ear.
    Taller, sharper, elegant point, slightly swept back.
    """
    return np.array([
        [ 0.00, -0.08],   # attach point
        [ 0.10,  0.02],
        [ 0.18,  0.18],
        [ 0.20,  0.38],
        [ 0.12,  0.62],
        [ 0.00,  0.86],   # main point
        [-0.10,  0.62],
        [-0.12,  0.34],
        [-0.08,  0.12],
    ])

def draw_single_filled_ear(ctx, base, side=1, template="round", ear_scale=1.0):
    """
    Draw one filled ear using a canned polygon template.
    """

    if template == "round":
        tmpl = make_round_ear_template()
        sx = 0.42 * ear_scale
        sy = 0.42 * ear_scale
        z = 11
    elif template == "ogre":
        tmpl = make_ogre_ear_template()
        sx = 0.50 * ear_scale
        sy = 0.50 * ear_scale
        z = 11
    else:  # goblin
        tmpl = make_goblin_ear_template()
        sx = 0.48 * ear_scale
        sy = 0.54 * ear_scale
        z = 11

    pts = transform_template_points(
        tmpl,
        base=base,
        side=side,
        sx=sx,
        sy=sy
    )

    ear = Polygon(
        pts,
        closed=True,
        facecolor=ctx.body_color,
        edgecolor="black",
        linewidth=1.5,
        zorder=z,
        joinstyle="round"
    )
    ctx.ax.add_patch(ear)

    # Small inner ear accent
    inner_pts = transform_template_points(
        tmpl * np.array([0.55, 0.55]),
        base=base,
        side=side,
        sx=sx,
        sy=sy,
        dx=0.01 * side * ear_scale,
        dy=0.04 * ear_scale
    )

    inner = Polygon(
        inner_pts,
        closed=False,
        fill=False,
        edgecolor="black",
        linewidth=0.9,
        alpha=0.35,
        zorder=z + 1,
        joinstyle="round"
    )
    ctx.ax.add_patch(inner)

    return ear

def draw_single_antanna(ctx, base, side=1, ear_scale=1.0):
    """
    Draw one antenna:
    - short stalk
    - 3 prong-like tips
    """
    bx, by = base

    stalk_len = 0.52 * ear_scale
    stalk_rise = 0.38 * ear_scale

    # Main elbow / tip direction
    mx = bx + side * 0.18 * stalk_len
    my = by + 0.45 * stalk_rise

    tx = bx + side * 0.55 * stalk_len
    ty = by + stalk_rise

    # Main stalk
    ctx.ax.plot(
        [bx, mx, tx],
        [by, my, ty],
        color="black",
        linewidth=2.2,
        zorder=11,
        solid_capstyle="round"
    )

    # Three little prongs
    prong_len = 0.12 * ear_scale
    prong_angles = [2.3, 1.75, 1.2]  # visually nice spread

    for ang in prong_angles:
        # flip horizontally for left side
        dx = side * prong_len * math.cos(ang)
        dy = prong_len * math.sin(ang)

        ctx.ax.plot(
            [tx, tx + dx],
            [ty, ty + dy],
            color="black",
            linewidth=2.0,
            zorder=12,
            solid_capstyle="round"
        )

    return {
        "base": base,
        "tip": (tx, ty)
    }

def draw_ears(ctx):
    """
    Draw ear traits using canned shapes + ctx-based placement.

    Supported:
    - antennae
    - ears
    - ogre
    - goblins
    """
    ear_type = ctx.v.get("ears", "n/a")

    if ear_type == "n/a":
        return None

    layout = get_ear_layout(ctx)
    left_base = layout["left_base"]
    right_base = layout["right_base"]
    scale = layout["scale"]

    if ear_type in ["antannae", "antanna"]:
        left = draw_single_antanna(ctx, left_base, side=-1, ear_scale=scale)
        right = draw_single_antanna(ctx, right_base, side=1, ear_scale=scale)

    elif ear_type in ["ears", "ear"]:
        left = draw_single_filled_ear(ctx, left_base, side=-1, template="round", ear_scale=scale)
        right = draw_single_filled_ear(ctx, right_base, side=1, template="round", ear_scale=scale)

    elif ear_type in ["ogre", "ogres"]:
        left = draw_single_filled_ear(ctx, left_base, side=-1, template="ogre", ear_scale=scale)
        right = draw_single_filled_ear(ctx, right_base, side=1, template="ogre", ear_scale=scale)

    elif ear_type in ["goblins", "goblin"]:
        left = draw_single_filled_ear(ctx, left_base, side=-1, template="goblin", ear_scale=scale)
        right = draw_single_filled_ear(ctx, right_base, side=1, template="goblin", ear_scale=scale)

    else:
        return None

    return {
        "type": ear_type,
        "left": left,
        "right": right,
        "layout": layout
    }

def color_luminance(rgb):
    """
    Approximate perceived luminance for an RGB tuple in [0,1].
    """
    r, g, b = rgb
    return 0.2126 * r + 0.7152 * g + 0.0722 * b

def mix_colors(c1, c2, t=0.5):
    """
    Linear blend between two RGB colors.
    t=0 -> c1
    t=1 -> c2
    """
    c1 = np.array(c1, dtype=float)
    c2 = np.array(c2, dtype=float)
    return tuple(np.clip((1 - t) * c1 + t * c2, 0, 1))

def adjust_color_brightness(color, factor=1.2):
    """
    Lighten/darken a matplotlib color.

    factor > 1 lightens
    factor < 1 darkens
    """
    try:
        rgb = np.array(mcolors.to_rgb(color))
    except Exception:
        rgb = np.array([0.1, 0.1, 0.1])

    if factor >= 1:
        rgb = rgb + (1 - rgb) * (factor - 1)
    else:
        rgb = rgb * factor

    return tuple(np.clip(rgb, 0, 1))

def rock_texture_is_curly(ctx):
    """
    True if this rock expresses curly hair texture.

    Supports phenotype strings and direct gene fallback.
    """
    texture = str(ctx.v.get("hair_texture", "n/a")).lower()

    if "curly" in texture:
        return True

    try:
        gene = str(ctx.rock.genes.get("hair_texture", "00"))
        return gene == "11"
    except Exception:
        return False

def deterministic_rng_for_rock(rock, salt="curl"):
    """
    Deterministic random generator so curls do not jump around every redraw.
    """
    seed_text = f"{getattr(rock, 'id', 0)}_{salt}_{str(getattr(rock, 'genes', {}))}"
    seed = abs(hash(seed_text)) % (2**32)
    return random.Random(seed)

def draw_curl_arc(
    ax,
    x,
    y,
    w,
    h,
    color,
    angle=0,
    theta1=0,
    theta2=180,
    linewidth=1.25,
    alpha=0.9,
    zorder=50
):
    """
    Draw one curl arc.
    """
    arc = Arc(
        (x, y),
        width=w,
        height=h,
        angle=angle,
        theta1=theta1,
        theta2=theta2,
        color=color,
        linewidth=linewidth,
        alpha=alpha,
        zorder=zorder
    )

    ax.add_patch(arc)

    return arc

def get_wrinkle_color(body_color):
    """
    Adaptive wrinkle color:
    - lighten dark rocks
    - darken light rocks

    This keeps wrinkle lines visible on all body colors.
    """
    lum = color_luminance(body_color)

    if lum < 0.45:
        # Dark body -> lighter wrinkles
        return mix_colors(body_color, (1, 1, 1), t=0.45)
    else:
        # Light body -> darker wrinkles
        return mix_colors(body_color, (0, 0, 0), t=0.35)

def get_wrinkle_y_fractions(ctx, n_lines=5):
    """
    Choose wrinkle bands across the body.
    Focus on the broad middle/interior regions.
    """
    shape = ctx.v.get("shape", "circle")

    if shape == "triangle":
        base = [np.random.uniform(0.1,0.9), np.random.uniform(0.1,0.9), np.random.uniform(0.1,0.9), np.random.uniform(0.1,0.9)]
    elif shape == "oblong":
        base = [np.random.uniform(0.1,0.9), np.random.uniform(0.1,0.9), np.random.uniform(0.1,0.9), np.random.uniform(0.1,0.9)]
    else:
        base = [np.random.uniform(0.1,0.9), np.random.uniform(0.1,0.9), np.random.uniform(0.1,0.9), np.random.uniform(0.1,0.9)]

    if n_lines <= len(base):
        return base[:n_lines]

    extra = list(np.linspace(0.9, 0.1, n_lines))
    return extra

def draw_wrinkles(ctx):
    """
    Draw surface wrinkle lines across the rock.

    Style:
    - irregular short/medium lines
    - body-aware span
    - clipped to body
    - adaptive color so visible on all body colors
    """

    wrinkle_type = ctx.v.get("wrinkles", "n/a")

    if wrinkle_type == "n/a":
        return None

    wrinkle_color = get_wrinkle_color(ctx.body_color)
    lines_drawn = []

    # Number of wrinkle rows.
    # If you later want stronger expression for double-alleles,
    # we can increase this based on allele count.
    y_fracs = get_wrinkle_y_fractions(ctx, n_lines=5)

    for idx, y_frac in enumerate(y_fracs):
        x_left, x_right, y = body_span_at_fraction(ctx, y_frac)
        local_width = x_right - x_left

        # Keep wrinkles a bit inside the edges.
        margin = 0.01 * local_width
        usable_left = x_left + margin
        usable_right = x_right - margin

        if usable_right <= usable_left:
            continue

        # Start point and total line length.
        total_len = ctx.py_rng.uniform(0.5 * local_width, 0.95 * local_width)
        start_x = ctx.py_rng.uniform(usable_left - total_len / 2 + local_width / 2, usable_left - total_len / 2 + local_width / 2 + 0.05 * local_width)

        # Build a small jagged / wavy wrinkle path.
        n_segments = ctx.py_rng.randint(4, 7)

        xs = [start_x]
        ys = [y + ctx.py_rng.uniform(-0.015, 0.015) * ctx.unit]

        current_x = start_x
        current_y = ys[0]

        for s in range(n_segments):
            dx = total_len / n_segments
            dy = ctx.py_rng.uniform(-0.06, 0.06) * ctx.unit

            # Sometimes make a more angular “step”.
            if s % 2 == 1 and ctx.py_rng.random() < 0.35:
                dy *= 0.4

            current_x += dx
            current_y += dy

            if current_x > usable_right:
                break

            xs.append(current_x)
            ys.append(current_y)

        line, = ctx.ax.plot(
            xs,
            ys,
            color=wrinkle_color,
            linewidth=1.6,
            alpha=0.95,
            zorder=4,
            solid_capstyle="round"
        )
        line.set_clip_path(ctx.body)
        lines_drawn.append(line)

    return lines_drawn

def get_freckle_color(body_color):
    """
    Adaptive freckle color.

    Freckles should look like small mineral inclusions.
    - on light rocks: darker freckles
    - on dark rocks: lighter freckles
    """
    lum = color_luminance(body_color)

    if lum < 0.42:
        return mix_colors(body_color, (1, 1, 1), t=0.55)
    else:
        return mix_colors(body_color, (0, 0, 0), t=0.45)

def random_point_in_body(ctx, max_attempts=100):
    """
    Sample a random point inside the body polygon.
    """
    body_path = Path(ctx.body_points)

    for _ in range(max_attempts):
        x = ctx.py_rng.uniform(ctx.xmin, ctx.xmax)
        y = ctx.py_rng.uniform(ctx.ymin, ctx.ymax)

        if body_path.contains_point((x, y)):
            return x, y

    # Fallback to center if sampling somehow fails.
    return ctx.cx, ctx.cy

def draw_freckles(ctx):
    """
    Draw freckles/mineral speckles on the body surface.

    Current expression:
    - if freckles phenotype is n/a, draw none
    - if freckles are active, draw scattered small dots

    Uses active allele count as intensity:
    - one active allele: fewer freckles
    - two active alleles: more freckles

    If your phenotype says freckles only express at 11, this still works.
    """
    freckle_trait = ctx.v.get("freckles", "n/a")

    if freckle_trait == "n/a":
        return None

    values = ctx.v.get("freckles_values", [])
    active_count = sum(1 for val in values if val != 0)

    # If somehow active_count is zero but phenotype says freckles, default to 1.
    active_count = max(1, active_count)

    # Number and size scale.
    if active_count == 1:
        n_freckles = 10
        r_min = 0.010 * ctx.unit
        r_max = 0.024 * ctx.unit
    else:
        n_freckles = 20
        r_min = 0.012 * ctx.unit
        r_max = 0.032 * ctx.unit

    freckle_color = get_freckle_color(ctx.body_color)

    freckles = []

    for _ in range(n_freckles):
        x, y = random_point_in_body(ctx)

        # Bias freckles slightly toward the visible/front middle,
        # so they do not all vanish near the silhouette.
        if ctx.py_rng.random() < 0.55:
            x = 0.72 * x + 0.28 * ctx.cx
            y = 0.80 * y + 0.20 * ctx.cy

        radius = ctx.py_rng.uniform(r_min, r_max)

        dot = Circle(
            (x, y),
            radius=radius,
            facecolor=freckle_color,
            edgecolor="none",
            alpha=0.80,
            zorder=5
        )

        dot.set_clip_path(ctx.body)
        ctx.ax.add_patch(dot)
        freckles.append(dot)

    return freckles

def get_arm_y_fractions(ctx, n_pairs):
    """
    Choose vertical attachment positions for arm pairs.

    The values are body-height fractions:
    0 = bottom of body
    1 = top of body

    We keep arms in the middle band to avoid eyes/mouth/hair/feet-space.
    """

    if n_pairs <= 0:
        return []

    shape = ctx.v.get("shape", "circle")

    if shape == "triangle":
        # Triangle is narrow high up, so arms attach slightly lower.
        if n_pairs == 1:
            return [0.50]
        if n_pairs == 2:
            return [0.50, 0.2]
        return np.linspace(0.5, 0.2, n_pairs)

    elif shape == "oblong":
        # Oblongs are wide, can support a clean mid-band.
        if n_pairs == 1:
            return [0.50]
        if n_pairs == 2:
            return [0.50, 0.2]
        return np.linspace(0.5, 0.2, n_pairs)

    elif shape == "oval":
        if n_pairs == 1:
            return [0.5]
        if n_pairs == 2:
            return [0.5, 0.2]
        return np.linspace(0.5, 0.2, n_pairs)

    elif shape == "square":
        if n_pairs == 1:
            return [0.5]
        if n_pairs == 2:
            return [0.5, 0.2]
        return np.linspace(0.5, 0.2, n_pairs)

    else:
        if n_pairs == 1:
            return [0.5]
        if n_pairs == 2:
            return [0.5, 0.2]
        return np.linspace(0.5, 0.2, n_pairs)

def get_arm_anchor_points(ctx, n_pairs):
    """
    Returns arm anchor points on the actual left and right body edges.

    Output:
    [
        {
            "left":  (x_left, y),
            "right": (x_right, y),
            "y_frac": y_frac,
            "span": x_right - x_left
        },
        ...
    ]
    """
    y_fracs = get_arm_y_fractions(ctx, n_pairs)
    anchors = []

    for y_frac in y_fracs:
        x_left, x_right, y = body_span_at_fraction(ctx, y_frac)

        anchors.append({
            "left": (x_left, y),
            "right": (x_right, y),
            "y_frac": y_frac,
            "span": x_right - x_left
        })

    return anchors

def draw_single_normal_arm(ctx, attach, side=1, layer_offset=0):
    """
    Draw one stick-style arm connected to the body edge.

    side = -1 for left, +1 for right
    """

    ax = ctx.ax
    x0, y0 = attach

    # Scale arm length to rock size but not wildly.
    arm_len = 0.42 * ctx.unit
    forearm_len = 0.28 * ctx.unit

    # Slight deterministic pose variation.
    bend = ctx.py_rng.uniform(-0.1, 0.1) * ctx.unit

    elbow_x = x0 + side * arm_len
    elbow_y = y0 + 0.10 * ctx.unit + bend

    hand_x = elbow_x + side * forearm_len
    hand_y = elbow_y - 0.16 * ctx.unit

    # Tiny shoulder dot at exact edge connection.
    ax.add_patch(
        Circle(
            (x0, y0),
            radius=0.030 * ctx.unit,
            facecolor=ctx.body_color,
            edgecolor="black",
            linewidth=1.0,
            zorder=3 + layer_offset
        )
    )

    # Upper arm and forearm.
    ax.plot(
        [x0, elbow_x],
        [y0, elbow_y],
        color="black",
        linewidth=2.0,
        solid_capstyle="round",
        zorder=2 + layer_offset
    )

    ax.plot(
        [elbow_x, hand_x],
        [elbow_y, hand_y],
        color="black",
        linewidth=2.0,
        solid_capstyle="round",
        zorder=2 + layer_offset
    )

    # Hand.
    ax.add_patch(
        Circle(
            (hand_x, hand_y),
            radius=0.055 * ctx.unit,
            facecolor=ctx.body_color,
            edgecolor="black",
            linewidth=1.0,
            zorder=3 + layer_offset
        )
    )

    return {
        "shoulder": (x0, y0),
        "elbow": (elbow_x, elbow_y),
        "hand": (hand_x, hand_y)
    }

def draw_single_muscle_arm(ctx, attach, side=1, layer_offset=0):
    """
    Draw one thicker muscle arm connected to the body edge.
    """

    ax = ctx.ax
    x0, y0 = attach

    upper_len = 0.42 * ctx.unit
    fore_len = 0.32 * ctx.unit

    elbow_x = x0 + side * upper_len
    elbow_y = y0 + 0.08 * ctx.unit

    hand_x = elbow_x + side * fore_len
    hand_y = elbow_y - 0.13 * ctx.unit

    # Shoulder connector.
    ax.add_patch(
        Circle(
            (x0, y0),
            radius=0.045 * ctx.unit,
            facecolor=ctx.body_color,
            edgecolor="black",
            linewidth=1.1,
            zorder=3 + layer_offset
        )
    )

    # Thick upper arm.
    ax.plot(
        [x0, elbow_x],
        [y0, elbow_y],
        color="black",
        linewidth=5,
        solid_capstyle="round",
        zorder=2 + layer_offset
    )

    # Bicep bulge.
    bicep_x = x0 + side * 0.58 * upper_len
    bicep_y = y0 + 0.05 * ctx.unit

    ax.add_patch(
        Ellipse(
            (bicep_x, bicep_y),
            width=0.22 * ctx.unit,
            height=0.12 * ctx.unit,
            angle=side * 15,
            facecolor=ctx.body_color,
            edgecolor="black",
            linewidth=1.1,
            zorder=3 + layer_offset
        )
    )

    # Thick forearm.
    ax.plot(
        [elbow_x, hand_x],
        [elbow_y, hand_y],
        color="black",
        linewidth=4,
        solid_capstyle="round",
        zorder=2 + layer_offset
    )

    # Fist.
    ax.add_patch(
        Ellipse(
            (hand_x, hand_y),
            width=0.18 * ctx.unit,
            height=0.14 * ctx.unit,
            angle=side * -12,
            facecolor=ctx.body_color,
            edgecolor="black",
            linewidth=1.1,
            zorder=3 + layer_offset
        )
    )

    return {
        "shoulder": (x0, y0),
        "elbow": (elbow_x, elbow_y),
        "hand": (hand_x, hand_y)
    }

def draw_arms(ctx):
    """
    Draw arms based on co-dominant arm alleles.

    arms gene:
    0 = no arm
    1 = normal arm pair
    2 = muscle arm pair

    Examples:
    00 -> no arms
    01 -> 2 normal arms
    11 -> 4 normal arms
    02 -> 2 muscle arms
    12 -> 2 normal arms + 2 muscle arms
    22 -> 4 muscle arms
    """

    normal_pairs = ctx.v.get("normal_arm_pairs", 0)
    muscle_pairs = ctx.v.get("muscle_arm_pairs", 0)

    total_pairs = normal_pairs + muscle_pairs

    if total_pairs <= 0:
        return []

    anchors = get_arm_anchor_points(ctx, total_pairs)

    drawn = []

    pair_types = []

    # Put muscle arms first so they usually appear slightly behind normal arms.
    for _ in range(muscle_pairs):
        pair_types.append("muscle")

    for _ in range(normal_pairs):
        pair_types.append("normal")

    for i, pair_type in enumerate(pair_types):
        anchor = anchors[i]

        left_attach = anchor["left"]
        right_attach = anchor["right"]

        # Lower pairs draw slightly behind upper pairs.
        layer_offset = i

        if pair_type == "muscle":
            drawn.append(draw_single_muscle_arm(ctx, left_attach, side=-1, layer_offset=layer_offset))
            drawn.append(draw_single_muscle_arm(ctx, right_attach, side=1, layer_offset=layer_offset))

        else:
            drawn.append(draw_single_normal_arm(ctx, left_attach, side=-1, layer_offset=layer_offset))
            drawn.append(draw_single_normal_arm(ctx, right_attach, side=1, layer_offset=layer_offset))

    return drawn

def get_crown_layout(ctx):
    """
    Compute a top-of-head anchor and scale for crowns.

    Returns a dict with:
    - center x
    - top y of the body
    - local width near the top
    - diamond size scale
    """
    crown_type = ctx.v.get("crowns", "n/a")

    if crown_type == "n/a":
        return None

    # Sample a band slightly below the very top to estimate width.
    x_left, x_right, y_band = body_span_at_fraction(ctx, 0.90)

    top_x = ctx.cx
    top_y = ctx.ymax

    local_width = max(1e-6, x_right - x_left)

    # Base diamond size from local width and overall body size.
    diamond_w = min(0.34 * local_width, 0.24 * ctx.unit)
    diamond_h = 0.18 * ctx.unit

    return {
        "type": crown_type,
        "cx": top_x,
        "top_y": top_y,
        "local_width": local_width,
        "diamond_w": diamond_w,
        "diamond_h": diamond_h,
        "x_left": x_left,
        "x_right": x_right,
        "y_band": y_band
    }

def make_diamond(cx, cy, w, h):
    """
    Return diamond polygon points centered at (cx, cy).
    """
    return [
        [cx, cy + h / 2],
        [cx + w / 2, cy],
        [cx, cy - h / 2],
        [cx - w / 2, cy],
    ]

def draw_crown(ctx):
    """
    Draw crown traits using ctx.

    Crown types:
    - small  -> one black diamond with gold outline
    - medium -> black / white stacked diamonds with gold outline
    - large  -> black / white / black stacked diamonds with gold outline
    - indent -> a divot cut into the top of the rock, ending at the rock edge
    """

    crown_type = ctx.v.get("crowns", "n/a")

    if crown_type == "n/a":
        return None

    layout = get_crown_layout(ctx)
    cx = layout["cx"]
    top_y = layout["top_y"]
    dw = layout["diamond_w"]
    dh = layout["diamond_h"]

    gold_edge = "gold"
    gold_lw = 1.0

    # -----------------------------
    # Indent crown: divot into head
    # -----------------------------
    if crown_type == "indent":
        bg = ctx.ax.get_facecolor()

        notch_w = 0.42 * ctx.unit
        notch_d = 0.18 * ctx.unit

        # Cut a notch exactly from the top edge downward.
        notch = Polygon(
            [
                [cx - notch_w / 2, top_y],
                [cx, top_y - notch_d],
                [cx + notch_w / 2, top_y],
            ],
            closed=True,
            facecolor=bg,
            edgecolor="none",
            zorder=15
        )
        notch.set_clip_path(ctx.body)
        ctx.ax.add_patch(notch)

        # Draw only the two sloped sides of the divot.
        # These begin and end exactly on the rock edge.
        ctx.ax.plot(
            [cx - notch_w / 2, cx],
            [top_y, top_y - notch_d],
            color="black",
            linewidth=2.1,
            zorder=16,
            clip_path=ctx.body
        )
        ctx.ax.plot(
            [cx, cx + notch_w / 2],
            [top_y - notch_d, top_y],
            color="black",
            linewidth=2.1,
            zorder=16,
            clip_path=ctx.body
        )

        return {
            "type": crown_type,
            "center": (cx, top_y),
        }

    # -----------------------------
    # Stacked diamond crowns
    # -----------------------------
    if crown_type == "small":
        stack_colors = ["black"]
    elif crown_type == "medium":
        stack_colors = ["black", "white"]
    elif crown_type == "large":
        stack_colors = ["black", "white", "black"]
    else:
        stack_colors = ["black"]

    # Bottom diamond sits slightly into the head, like your sketch.
    base_cy = top_y + 0.04 * ctx.unit

    drawn = []

    for i, fill_color in enumerate(stack_colors):
        cy = base_cy + i * (0.58 * dh)

        # Slight taper up the stack.
        scale = 1.00 - 0.06 * i
        w = dw * scale
        h = dh * scale

        diamond = Polygon(
            make_diamond(cx, cy, w, h),
            closed=True,
            facecolor=fill_color,
            edgecolor=gold_edge,
            linewidth=gold_lw,
            zorder=15 + i,
            joinstyle="miter"
        )
        ctx.ax.add_patch(diamond)
        drawn.append(diamond)

    return {
        "type": crown_type,
        "center": (cx, top_y),
        "count": len(stack_colors)
    }

def get_eye_layout(ctx, eye_count):
    """
    Returns eye positions and eye radius that respect body shape and size.
    """
    shape = ctx.v.get("shape", "circle")

    # Triangles are narrow near the top, so eyes sit lower.
    # Oblong rocks can support wider-set eyes.
    if shape == "triangle":
        y_frac = 0.50
        spread_factor = 0.28
    elif shape == "oblong":
        y_frac = 0.58
        spread_factor = 0.34
    elif shape == "oval":
        y_frac = 0.58
        spread_factor = 0.28
    elif shape == "square":
        y_frac = 0.58
        spread_factor = 0.30
    else:
        y_frac = 0.58
        spread_factor = 0.30

    x_left, x_right, y = body_span_at_fraction(ctx, y_frac)

    available_width = x_right - x_left
    center_x = 0.5 * (x_left + x_right)

    # Eye radius scales with local rock size.
    eye_radius = min(
        0.085 * ctx.height,
        0.13 * available_width
    )

    eye_radius = max(eye_radius, 0.045 * ctx.unit)

    margin = 1.35 * eye_radius

    if eye_count <= 0:
        return [], eye_radius

    if eye_count == 1:
        return [(center_x, y)], eye_radius * 1.08

    # Two eyes.
    separation = available_width * spread_factor

    left_x = clamp_inside_span(center_x - separation, x_left, x_right, margin)
    right_x = clamp_inside_span(center_x + separation, x_left, x_right, margin)

    return [(left_x, y), (right_x, y)], eye_radius

def get_eye_color(color_name):
    return EYE_COLOR_MAP.get(color_name, "white")

def draw_eyes(ctx):
    """
    Draw eyes using the new shape-aware layout.
    """
    eye_count = ctx.v.get("eyes_count", 0)

    eye_positions, eye_radius = get_eye_layout(ctx, eye_count)

    if len(eye_positions) == 0:
        return []

    eye_color_name = ctx.v.get("eye_color", "black")
    eye_color = get_eye_color(eye_color_name)

    drawn_positions = []

    for ex, ey in eye_positions:
        sclera_color = eye_color #"white"

        if eye_color_name == "callus":
            sclera_color = "tan"

        eye = Circle(
            (ex, ey),
            radius=eye_radius,
            facecolor=sclera_color,
            edgecolor="black",
            linewidth=1.2,
            zorder=8
        )
        ctx.ax.add_patch(eye)

        pupil = Circle(
            (ex, ey),
            radius=0.43 * eye_radius,
            facecolor=eye_color,
            edgecolor="black",
            linewidth=0.5,
            zorder=9
        )
        ctx.ax.add_patch(pupil)

        # Optional evil eye accent.
        if eye_color_name == "evil":
            ctx.ax.plot(
                [ex, ex],
                [ey + 0.45 * eye_radius, ey - 0.45 * eye_radius],
                color="black",
                linewidth=1.4,
                zorder=10
            )

        drawn_positions.append((ex, ey, eye_radius))

    return drawn_positions

def draw_brows(ctx, drawn_eye_positions):
    """
    Draw brows using eye positions from draw_eyes(ctx).

    Supports:
    - brows
    - eyehair
    - unibrows

    Uses ctx so brows respect shape, size, and actual eye placement.
    """

    brow_type = ctx.v.get("brows", "n/a")

    if brow_type == "n/a":
        return

    # Brows need eyes. If there are no eyes, skip for now.
    # Later we can make "orphan brows" into a cursed rare visual.
    if len(drawn_eye_positions) == 0:
        return

    hair_color = get_hair_color_from_alleles(
        ctx.v.get("hair_color_alleles", [ctx.v.get("hair_color", "black")])
    )

    curly = ctx.v.get("hair_texture", "straight") == "curly"

    # Use eye radius as local scale.
    avg_eye_radius = sum(r for _, _, r in drawn_eye_positions) / len(drawn_eye_positions)

    # Brow y-position sits above the eye.
    brow_lift = 1.35 * avg_eye_radius

    # Body slice near brow height for clamping.
    # Use the first eye to estimate face band.
    sample_eye_y = drawn_eye_positions[0][1]
    y_frac = (sample_eye_y - ctx.ymin) / max(ctx.height, 1e-9)
    brow_frac = min(0.92, y_frac + 0.12)

    x_left, x_right, brow_band_y = body_span_at_fraction(ctx, brow_frac)

    def clamp_brow_x(x, margin):
        return clamp_inside_span(x, x_left, x_right, margin)

    # -----------------------------
    # Normal separate brows
    # -----------------------------

    if brow_type == "brows":
        for ex, ey, er in drawn_eye_positions:
            brow_y = ey + brow_lift
            brow_half_width = 1.05 * er

            x0 = clamp_brow_x(ex - brow_half_width, 0.20 * er)
            x1 = clamp_brow_x(ex + brow_half_width, 0.20 * er)

            # Slight angry/expressive slant.
            if ex < ctx.cx:
                y0 = brow_y + 0.12 * er
                y1 = brow_y + 0.28 * er
            elif ex > ctx.cx:
                y0 = brow_y + 0.28 * er
                y1 = brow_y + 0.12 * er
            else:
                y0 = brow_y + 0.18 * er
                y1 = brow_y + 0.18 * er

            if curly:
                arc = Arc(
                    (ex, brow_y + 0.12 * er),
                    2.2 * er,
                    0.9 * er,
                    theta1=20,
                    theta2=160,
                    color=hair_color,
                    linewidth=2.2,
                    zorder=10
                )
                ctx.ax.add_patch(arc)
            else:
                ctx.ax.plot(
                    [x0, x1],
                    [y0, y1],
                    color=hair_color,
                    linewidth=2.4,
                    solid_capstyle="round",
                    zorder=10
                )

    # -----------------------------
    # Eyehair: little lashes/tufts over each eye
    # -----------------------------

    elif brow_type == "eyehair":
        for ex, ey, er in drawn_eye_positions:
            n_hairs = 5
            spread = 1.45 * er

            for k in range(n_hairs):
                t = 0 if n_hairs == 1 else k / (n_hairs - 1)
                hx = ex - spread / 2 + t * spread
                hy = ey + 1.15 * er

                hx = clamp_brow_x(hx, 0.15 * er)

                if curly:
                    arc = Arc(
                        (hx, hy + 0.32 * er),
                        0.50 * er,
                        0.55 * er,
                        theta1=0,
                        theta2=300,
                        color=hair_color,
                        linewidth=1.4,
                        zorder=10
                    )
                    ctx.ax.add_patch(arc)
                else:
                    lean = ctx.py_rng.uniform(-0.25, 0.25) * er
                    length = ctx.py_rng.uniform(0.65, 1.05) * er

                    ctx.ax.plot(
                        [hx, hx + lean],
                        [hy, hy + length],
                        color=hair_color,
                        linewidth=1.3,
                        solid_capstyle="round",
                        zorder=10
                    )

    # -----------------------------
    # Unibrow: one connected brow across eye region
    # -----------------------------

    elif brow_type == "unibrows":
        eye_xs = [x for x, y, r in drawn_eye_positions]
        eye_ys = [y for x, y, r in drawn_eye_positions]

        center_y = sum(eye_ys) / len(eye_ys)
        brow_y = center_y + brow_lift

        if len(drawn_eye_positions) == 1:
            # Single-eye unibrow becomes a thick brow over the cyclops eye.
            ex, ey, er = drawn_eye_positions[0]
            x0 = ex - 1.65 * er
            x1 = ex + 1.65 * er
        else:
            left_eye = min(eye_xs)
            right_eye = max(eye_xs)
            x0 = left_eye - 1.1 * avg_eye_radius
            x1 = right_eye + 1.1 * avg_eye_radius

        x0 = clamp_brow_x(x0, 0.2 * avg_eye_radius)
        x1 = clamp_brow_x(x1, 0.2 * avg_eye_radius)

        if curly:
            n_curls = 5
            for k in range(n_curls):
                t = 0 if n_curls == 1 else k / (n_curls - 1)
                cx = x0 + t * (x1 - x0)

                arc = Arc(
                    (cx, brow_y),
                    0.55 * avg_eye_radius,
                    0.50 * avg_eye_radius,
                    theta1=0,
                    theta2=320,
                    color=hair_color,
                    linewidth=2.0,
                    zorder=10
                )
                ctx.ax.add_patch(arc)
        else:
            xs = np.linspace(x0, x1, 80)
            ys = brow_y + 0.10 * avg_eye_radius * np.sin(
                np.linspace(0, 2 * np.pi, 80)
            )

            ctx.ax.plot(
                xs,
                ys,
                color=hair_color,
                linewidth=3.0,
                solid_capstyle="round",
                zorder=10
            )

def get_nose_layout(ctx, drawn_eye_positions=None):
    """
    Compute a nose position between eyes and mouth.

    Goals:
    - Stay below eyes
    - Stay above mouth
    - Stay inside the body span
    - Leave mouth/facial-hair space clean
    """

    shape = ctx.v.get("shape", "circle")
    nose_type = ctx.v.get("noses", "n/a")

    # Estimate mouth position without drawing it.
    mouth_cx, mouth_cy, mouth_w, mouth_h, mouth_x_left, mouth_x_right = get_mouth_layout(
        ctx,
        drawn_eye_positions=drawn_eye_positions
    )

    # Estimate eye lower bound.
    if drawn_eye_positions is not None and len(drawn_eye_positions) > 0:
        avg_eye_y = sum(y for x, y, r in drawn_eye_positions) / len(drawn_eye_positions)
        avg_eye_r = sum(r for x, y, r in drawn_eye_positions) / len(drawn_eye_positions)

        eye_bottom = avg_eye_y - 1.10 * avg_eye_r
        eye_center_y = avg_eye_y
    else:
        avg_eye_r = 0.07 * ctx.unit
        eye_center_y = ctx.ymin + 0.58 * ctx.height
        eye_bottom = eye_center_y - avg_eye_r

    # Estimate mouth top. Smiles/chips/smeagol occupy a little more vertical room.
    if ctx.v.get("mouths", "n/a") in ["smile", "chip", "smeagol"]:
        mouth_top = mouth_cy + 0.35 * mouth_h
    else:
        mouth_top = mouth_cy + 0.15 * mouth_h

    # Available vertical lane between eyes and mouth.
    top_limit = eye_bottom - 0.08 * ctx.unit
    bottom_limit = mouth_top + 0.10 * ctx.unit

    # If there is room, place nose in the lane.
    if top_limit > bottom_limit:
        nose_y = 0.56 * top_limit + 0.44 * bottom_limit
    else:
        # Emergency fallback: squeeze nose around body mid-face.
        nose_y = ctx.ymin + 0.47 * ctx.height

    # Shape-specific correction.
    # Triangles get narrow near the top, so nose sits lower.
    if shape == "triangle":
        nose_y = min(nose_y, ctx.ymin + 0.49 * ctx.height)
        nose_y = max(nose_y, ctx.ymin + 0.35 * ctx.height)
    elif shape == "oblong":
        nose_y = min(nose_y, ctx.ymin + 0.55 * ctx.height)
        nose_y = max(nose_y, ctx.ymin + 0.38 * ctx.height)
    else:
        nose_y = min(nose_y, ctx.ymin + 0.56 * ctx.height)
        nose_y = max(nose_y, ctx.ymin + 0.36 * ctx.height)

    # Get body width at chosen nose height.
    y_frac = (nose_y - ctx.ymin) / max(ctx.height, 1e-9)
    x_left, x_right, nose_y = body_span_at_fraction(ctx, y_frac)

    local_width = x_right - x_left
    center_x = 0.5 * (x_left + x_right)

    # Nose base scale. Different nose styles get different sizes.
    base_w = 0.13 * local_width
    base_h = 0.10 * ctx.height

    if nose_type == "nub":
        nw = 0.85 * base_w
        nh = 0.85 * base_h
    elif nose_type == "honk":
        nw = 1.55 * base_w
        nh = 1.15 * base_h
    elif nose_type == "holes":
        nw = 1.10 * base_w
        nh = 0.80 * base_h
    elif nose_type == "concave":
        nw = 1.65 * base_w
        nh = 1.00 * base_h
    else:
        nw = base_w
        nh = base_h

    # Clamp to sane limits.
    nw = max(0.08 * ctx.unit, min(nw, 0.30 * local_width))
    nh = max(0.05 * ctx.unit, min(nh, 0.18 * ctx.height))

    return {
        "type": nose_type,
        "center": (center_x, nose_y),
        "width": nw,
        "height": nh,
        "x_left": x_left,
        "x_right": x_right,
        "mouth_layout": {
            "center": (mouth_cx, mouth_cy),
            "width": mouth_w,
            "height": mouth_h,
            "x_left": mouth_x_left,
            "x_right": mouth_x_right,
        }
    }

def draw_nose(ctx, drawn_eye_positions=None):
    """
    Draw noses with ctx-based placement.

    Supported:
    - nub
    - honk
    - holes
    - concave
    """

    nose_type = ctx.v.get("noses", "n/a")

    if nose_type == "n/a":
        return None

    layout = get_nose_layout(ctx, drawn_eye_positions=drawn_eye_positions)

    cx, cy = layout["center"]
    nw = layout["width"]
    nh = layout["height"]

    # Use body color for protruding nose types.
    nose_fill = ctx.body_color

    # -----------------------------
    # Nub: small round bump
    # -----------------------------
    if nose_type == "nub":
        nose = Circle(
            (cx, cy),
            radius=0.50 * min(nw, nh),
            facecolor=nose_fill,
            edgecolor="black",
            linewidth=1.1,
            zorder=10
        )
        nose.set_clip_path(ctx.body)
        ctx.ax.add_patch(nose)

        # Tiny highlight
        ctx.ax.add_patch(
            Circle(
                (cx - 0.15 * nw, cy + 0.12 * nh),
                radius=0.10 * min(nw, nh),
                facecolor="white",
                edgecolor="none",
                alpha=0.35,
                zorder=11
            )
        )

    # -----------------------------
    # Honk: big goofy oval nose
    # -----------------------------
    elif nose_type == "honk":
        nose = Ellipse(
            (cx, cy),
            width=nw,
            height=nh,
            facecolor=nose_fill,
            edgecolor="black",
            linewidth=1.2,
            zorder=10
        )
        nose.set_clip_path(ctx.body)
        ctx.ax.add_patch(nose)

        # Nostril dot
        ctx.ax.add_patch(
            Circle(
                (cx + 0.20 * nw, cy - 0.02 * nh),
                radius=0.08 * min(nw, nh),
                facecolor="black",
                edgecolor="none",
                zorder=11
            )
        )

    # -----------------------------
    # Holes: two nostril holes only
    # -----------------------------
    elif nose_type == "holes":
        hole_r = 0.17 * min(nw, nh)

        for side in [-1, 1]:
            hole = Circle(
                (cx + side * 0.25 * nw, cy),
                radius=hole_r,
                facecolor="black",
                edgecolor="none",
                zorder=10
            )
            hole.set_clip_path(ctx.body)
            ctx.ax.add_patch(hole)

    # -----------------------------
    # Concave: inward curved nose/notch
    # -----------------------------
    elif nose_type == "concave":
        # A downward-facing shallow arc reads like a dent.
        arc = Arc(
            (cx, cy + 0.08 * nh),
            width=nw,
            height=nh,
            theta1=200,
            theta2=340,
            color="black",
            linewidth=1.8,
            zorder=10
        )
        arc.set_clip_path(ctx.body)
        ctx.ax.add_patch(arc)

        # Small shadow mark under it
        ctx.ax.plot(
            [cx - 0.20 * nw, cx + 0.20 * nw],
            [cy - 0.18 * nh, cy - 0.12 * nh],
            color="black",
            alpha=0.25,
            linewidth=1.0,
            zorder=9,
            clip_path=ctx.body
        )

    return layout

def get_mouth_layout(ctx, drawn_eye_positions=None):
    """
    Compute a shape-aware mouth position and size.

    Returns:
    center_x, center_y, mouth_width, mouth_height, x_left, x_right
    """

    shape = ctx.v.get("shape", "circle")

    # If eyes exist, place mouth below them.
    if drawn_eye_positions is not None and len(drawn_eye_positions) > 0:
        avg_eye_y = sum(y for x, y, r in drawn_eye_positions) / len(drawn_eye_positions)
        avg_eye_r = sum(r for x, y, r in drawn_eye_positions) / len(drawn_eye_positions)

        # Mouth sits below eyes.
        desired_y = avg_eye_y - 2.25 * avg_eye_r

        # Convert desired y to body fraction.
        y_frac = (desired_y - ctx.ymin) / max(ctx.height, 1e-9)

    else:
        # Fallback if the rock has no eyes.
        if shape == "triangle":
            y_frac = 0.34
        elif shape == "oblong":
            y_frac = 0.42
        else:
            y_frac = 0.38

    # Keep mouth inside a reasonable lower-face band.
    if shape == "triangle":
        y_frac = max(0.25, min(0.48, y_frac))
    else:
        y_frac = max(0.26, min(0.48, y_frac))

    x_left, x_right, y = body_span_at_fraction(ctx, y_frac)

    local_width = x_right - x_left
    center_x = 0.5 * (x_left + x_right)

    # Mouth size scales with local body span.
    mouth_width = 0.36 * local_width
    mouth_height = 0.13 * ctx.height

    # Avoid absurdly small or giant mouths.
    mouth_width = max(0.18 * ctx.unit, min(mouth_width, 0.58 * ctx.unit))
    mouth_height = max(0.06 * ctx.unit, min(mouth_height, 0.18 * ctx.unit))

    return center_x, y, mouth_width, mouth_height, x_left, x_right

def draw_mouth(ctx, drawn_eye_positions=None):
    """
    Shape-aware mouth drawer, style v2.

    mouth:
        simple neutral line

    smile:
        large orange-slice toothy grin

    chip:
        Patrick-style curved smile with one big off-center tooth

    smeagol:
        cursed jagged-tooth grin
    """

    mouth_type = ctx.v.get("mouths", "n/a")

    if mouth_type == "n/a":
        return None

    cx, cy, mw, mh, x_left, x_right = get_mouth_layout(
        ctx,
        drawn_eye_positions=drawn_eye_positions
    )

    # Make cartoon mouths more readable.
    if mouth_type in ["smile", "chip", "smeagol"]:
        mw *= 1.25
        mh *= 1.35

    line_color = "black"

    # Keep mouth width inside the body span.
    max_width = 0.86 * (x_right - x_left)
    mw = min(mw, max_width)

    # -----------------------------
    # Basic neutral mouth
    # -----------------------------
    if mouth_type == "mouth":
        x0 = cx - 0.50 * mw
        x1 = cx + 0.50 * mw

        x0 = clamp_inside_span(x0, x_left, x_right, 0.04 * ctx.unit)
        x1 = clamp_inside_span(x1, x_left, x_right, 0.04 * ctx.unit)

        ctx.ax.plot(
            [x0, x1],
            [cy, cy],
            color=line_color,
            linewidth=2.1,
            solid_capstyle="round",
            zorder=10
        )

    # -----------------------------
    # Smile: orange-slice toothy grin
    # -----------------------------
    elif mouth_type == "smile":
        left = cx - 0.50 * mw
        right = cx + 0.50 * mw
        top_y = cy + 0.12 * mh
        bottom_y = cy - 0.58 * mh

        left = clamp_inside_span(left, x_left, x_right, 0.04 * ctx.unit)
        right = clamp_inside_span(right, x_left, x_right, 0.04 * ctx.unit)

        # Orange-slice / crescent mouth shape.
        verts = [
            (left, top_y),
            (right, top_y),
            (right - 0.08 * mw, cy - 0.40 * mh),
            (cx, bottom_y),
            (left + 0.08 * mw, cy - 0.40 * mh),
            (left, top_y),
        ]

        codes = [
            Path.MOVETO,
            Path.LINETO,
            Path.CURVE3,
            Path.CURVE3,
            Path.CURVE3,
            Path.CURVE3,
        ]

        mouth_patch = PathPatch(
            Path(verts, codes),
            facecolor="white",
            edgecolor="black",
            linewidth=2.2,
            zorder=10
        )

        mouth_patch.set_clip_path(ctx.body)
        ctx.ax.add_patch(mouth_patch)

        # Tooth grid clipped inside the smile.
        n_vertical = 4

        for i in range(1, n_vertical):
            tx = left + i * (right - left) / n_vertical

            ctx.ax.plot(
                [tx, tx],
                [top_y, bottom_y + 0.10 * mh],
                color="black",
                linewidth=1.1,
                zorder=11,
                clip_path=mouth_patch
            )

        # One curved-ish horizontal tooth separator.
        xs = np.linspace(left + 0.05 * mw, right - 0.05 * mw, 80)
        ys = cy - 0.22 * mh + 0.04 * mh * np.cos(np.linspace(0, np.pi, 80))

        ctx.ax.plot(
            xs,
            ys,
            color="black",
            linewidth=1.1,
            zorder=11,
            clip_path=mouth_patch
        )

    # -----------------------------
    # Chip: Patrick smile with one big off-center tooth
    # -----------------------------
    elif mouth_type == "chip":
        left = cx - 0.46 * mw
        right = cx + 0.46 * mw
        top_y = cy + 0.05 * mh
        bottom_y = cy - 0.48 * mh

        left = clamp_inside_span(left, x_left, x_right, 0.04 * ctx.unit)
        right = clamp_inside_span(right, x_left, x_right, 0.04 * ctx.unit)

        # Big bowl smile.
        verts = [
            (left, top_y),
            (right, top_y),
            (right, cy - 0.38 * mh),
            (cx, bottom_y),
            (left, cy - 0.38 * mh),
            (left, top_y),
        ]

        codes = [
            Path.MOVETO,
            Path.LINETO,
            Path.CURVE3,
            Path.CURVE3,
            Path.CURVE3,
            Path.CURVE3,
        ]

        smile_patch = PathPatch(
            Path(verts, codes),
            facecolor="none",
            edgecolor="black",
            linewidth=2.2,
            zorder=10
        )

        smile_patch.set_clip_path(ctx.body)
        ctx.ax.add_patch(smile_patch)

        # One big off-center tooth hanging from the top line.
        tooth_cx = cx + 0.13 * mw
        tooth_top = top_y - 0.02 * mh
        tooth_bottom = cy - 0.30 * mh
        tooth_half_w = 0.105 * mw

        tooth = Polygon(
            [
                [tooth_cx - tooth_half_w, tooth_top],
                [tooth_cx + tooth_half_w, tooth_top],
                [tooth_cx + 0.72 * tooth_half_w, tooth_bottom],
                [tooth_cx - 0.72 * tooth_half_w, tooth_bottom],
            ],
            closed=True,
            facecolor="white",
            edgecolor="black",
            linewidth=1.1,
            zorder=11
        )

        tooth.set_clip_path(ctx.body)
        ctx.ax.add_patch(tooth)

    # -----------------------------
    # Smeagol: jagged chip smile with 2-3 teeth
    # -----------------------------
    elif mouth_type == "smeagol":
        left = cx - 0.50 * mw
        right = cx + 0.48 * mw
        top_y = cy + 0.04 * mh
        bottom_y = cy - 0.45 * mh

        left = clamp_inside_span(left, x_left, x_right, 0.04 * ctx.unit)
        right = clamp_inside_span(right, x_left, x_right, 0.04 * ctx.unit)

        # Uneven bowl-like smile.
        verts = [
            (left, top_y),
            (right, top_y + 0.03 * mh),
            (right - 0.02 * mw, cy - 0.34 * mh),
            (cx + 0.08 * mw, bottom_y),
            (left + 0.04 * mw, cy - 0.33 * mh),
            (left, top_y),
        ]

        codes = [
            Path.MOVETO,
            Path.LINETO,
            Path.CURVE3,
            Path.CURVE3,
            Path.CURVE3,
            Path.CURVE3,
        ]

        mouth_patch = PathPatch(
            Path(verts, codes),
            facecolor="none",
            edgecolor="black",
            linewidth=2.0,
            zorder=10
        )

        mouth_patch.set_clip_path(ctx.body)
        ctx.ax.add_patch(mouth_patch)

        # Jagged teeth.
        tooth_count = 2 + (ctx.rock.id % 2)

        tooth_centers = np.linspace(
            cx - 0.18 * mw,
            cx + 0.22 * mw,
            tooth_count
        )

        for i, tx in enumerate(tooth_centers):
            tooth_height = (0.27 + 0.08 * ((i + ctx.rock.id) % 2)) * mh
            tooth_width = 0.075 * mw

            tooth = Polygon(
                [
                    [tx - tooth_width, top_y - 0.02 * mh],
                    [tx + tooth_width, top_y - 0.02 * mh],
                    [tx + ctx.py_rng.uniform(-0.04, 0.04) * mw, top_y - tooth_height],
                ],
                closed=True,
                facecolor="white",
                edgecolor="black",
                linewidth=1.0,
                zorder=11
            )

            tooth.set_clip_path(ctx.body)
            ctx.ax.add_patch(tooth)

        # Little cursed wrinkles below mouth.
        for i in range(2):
            yy = cy - (0.50 + 0.20 * i) * mh
            ctx.ax.plot(
                [cx - 0.22 * mw, cx + 0.18 * mw],
                [yy, yy + 0.04 * mh * ((-1) ** i)],
                color="black",
                linewidth=0.9,
                alpha=0.45,
                zorder=9,
                clip_path=ctx.body
            )

    return {
        "type": mouth_type,
        "center": (cx, cy),
        "width": mw,
        "height": mh
    }

def get_facial_hair_layout(ctx, drawn_eye_positions=None, nose_info=None, mouth_info=None):
    """
    Compute safe layout bands for facial hair.

    We use:
    - nose info
    - mouth info
    - body spans at relevant heights

    So facial hair avoids overlapping the nose and mouth too aggressively.
    """
    fh_type = ctx.v.get("facial_hair", "n/a")

    if fh_type == "n/a":
        return None

    # If not supplied, estimate them.
    if mouth_info is None:
        mcx, mcy, mw, mh, mxl, mxr = get_mouth_layout(ctx, drawn_eye_positions)
        mouth_info = {
            "center": (mcx, mcy),
            "width": mw,
            "height": mh,
            "x_left": mxl,
            "x_right": mxr
        }

    if nose_info is None:
        nose_info = get_nose_layout(ctx, drawn_eye_positions)

    mouth_cx, mouth_cy = mouth_info["center"]
    mouth_w = mouth_info["width"]
    mouth_h = mouth_info["height"]

    nose_cx, nose_cy = nose_info["center"]
    nose_w = nose_info["width"]
    nose_h = nose_info["height"]

    # Useful vertical reference levels
    nose_bottom = nose_cy - 0.50 * nose_h
    mouth_top = mouth_cy + 0.18 * mouth_h
    mouth_bottom = mouth_cy - 0.18 * mouth_h

    # Mustache / stubble band between nose and mouth
    upper_band_y = 0.55 * nose_bottom + 0.45 * mouth_top

    # Patch / goatee / beard band below mouth
    lower_band_y = mouth_cy - 0.35 * mouth_h
    chin_band_y = mouth_cy - 0.75 * mouth_h

    # Clamp body-safe heights
    def clamp_y(y):
        return max(ctx.ymin + 0.08 * ctx.height, min(ctx.ymax - 0.08 * ctx.height, y))

    upper_band_y = clamp_y(upper_band_y)
    lower_band_y = clamp_y(lower_band_y)
    chin_band_y = clamp_y(chin_band_y)

    # Get local spans at these bands
    def span_at_y(y):
        y_frac = (y - ctx.ymin) / max(ctx.height, 1e-9)
        x_left, x_right, yy = body_span_at_fraction(ctx, y_frac)
        return x_left, x_right, yy

    upper_left, upper_right, upper_y = span_at_y(upper_band_y)
    lower_left, lower_right, lower_y = span_at_y(lower_band_y)
    chin_left, chin_right, chin_y = span_at_y(chin_band_y)

    return {
        "type": fh_type,
        "mouth_info": mouth_info,
        "nose_info": nose_info,

        "upper_band": {
            "y": upper_y,
            "x_left": upper_left,
            "x_right": upper_right
        },
        "lower_band": {
            "y": lower_y,
            "x_left": lower_left,
            "x_right": lower_right
        },
        "chin_band": {
            "y": chin_y,
            "x_left": chin_left,
            "x_right": chin_right
        }
    }

def draw_facial_hair_curls(
    ctx,
    hair_color,
    mouth_cx,
    mouth_cy,
    mouth_w,
    mouth_h,
    fh_type,
    zorder=62
):
    """
    Add curl marks over facial hair if hair_texture is curly.

    Works best for beard/goatee/curly styles.
    """
    if not rock_texture_is_curly(ctx):
        return []

    if fh_type in ["n/a", "peach_fuzz"]:
        return []

    # Small facial hair styles get fewer curls.
    if fh_type in ["beard"]:
        x_min = mouth_cx - 0.75 * mouth_w
        x_max = mouth_cx + 0.75 * mouth_w
        y_min = mouth_cy - 2.05 * mouth_h
        y_max = mouth_cy + 0.15 * mouth_h
        n_curls = 9
        curl_scale = 0.05

    elif fh_type in ["goatee"]:
        x_min = mouth_cx - 0.62 * mouth_w
        x_max = mouth_cx + 0.62 * mouth_w
        y_min = mouth_cy - 1.20 * mouth_h
        y_max = mouth_cy + 0.10 * mouth_h
        n_curls = 6
        curl_scale = 0.05

    elif fh_type in ["curly_mustache"]:
        x_min = mouth_cx - 0.65 * mouth_w
        x_max = mouth_cx + 0.65 * mouth_w
        y_min = mouth_cy - 0.15 * mouth_h
        y_max = mouth_cy + 0.35 * mouth_h
        n_curls = 5
        curl_scale = 0.05

    else:
        x_min = mouth_cx - 0.50 * mouth_w
        x_max = mouth_cx + 0.50 * mouth_w
        y_min = mouth_cy - 0.65 * mouth_h
        y_max = mouth_cy + 0.15 * mouth_h
        n_curls = 4
        curl_scale = 0.05

    return draw_curly_overlay_in_box(
        ctx,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        hair_color=hair_color,
        n_curls=n_curls,
        curl_scale=curl_scale,
        zorder=zorder,
        salt=f"facial_curls_{fh_type}"
    )

def draw_facial_hair(ctx, rock, v, drawn_eye_positions=None, nose_info=None, mouth_info=None):
    """
    Draw facial-hair styles:
    - goatee
    - beard
    - pedo
    - curl
    - chapman
    - sol
    """
    fh_type = ctx.v.get("facial_hair", "n/a")

    if fh_type == "n/a":
        return None

    layout = get_facial_hair_layout(
        ctx,
        drawn_eye_positions=drawn_eye_positions,
        nose_info=nose_info,
        mouth_info=mouth_info
    )

    mouth_info = layout["mouth_info"]
    nose_info = layout["nose_info"]

    mouth_cx, mouth_cy = mouth_info["center"]
    mouth_w = mouth_info["width"]
    mouth_h = mouth_info["height"]

    nose_cx, nose_cy = nose_info["center"]
    nose_w = nose_info["width"]
    nose_h = nose_info["height"]

    hair_color = get_hair_color_from_alleles(
        ctx.v.get("hair_color_alleles", [ctx.v.get("hair_color", "black")])
    )

    gender = v["gender"]

    z = 10

    # --------------------------------------------------
    # 0) PEACH FUZZ — soft small fuzz for females
    # --------------------------------------------------
        # --------------------------------------------------
    # 0) PEACH FUZZ — tiny fuzz spikes at ~7:30 and ~4:30
    # --------------------------------------------------
    if fh_type == "peach fuzz":
        fh_z = z - 1

        # Keep it subtle.
        fuzz_color = hair_color

        # Put the fuzz clusters slightly below mouth center,
        # out toward the lower-left and lower-right "cheek" areas.
        cluster_y = mouth_cy - 0.10 * mouth_h
        y_frac = (cluster_y - ctx.ymin) / max(ctx.height, 1e-9)
        x_left, x_right, cluster_y = body_span_at_fraction(ctx, y_frac)
        local_span = x_right - x_left

        # Anchor the fuzz on the face, not at the extreme edge.
        left_anchor = (
            x_left + 0.28 * local_span,
            cluster_y - 0.01 * ctx.unit
        )
        right_anchor = (
            x_right - 0.28 * local_span,
            cluster_y - 0.01 * ctx.unit
        )

        # 7:30 and 4:30 directions
        left_angle = math.radians(225)   # down-left
        right_angle = math.radians(-45)  # down-right

        cluster_specs = [
            (left_anchor, left_angle),
            (right_anchor, right_angle),
        ]

        for (ax0, ay0), base_angle in cluster_specs:
            n_spikes = 5

            for i in range(n_spikes):
                # Spread the spikes in a tiny fan
                ang = base_angle + np.linspace(-0.28, 0.28, n_spikes)[i]

                # Slightly offset each spike base so the cluster looks natural
                perp = base_angle + math.pi / 2
                base_offset = (i - (n_spikes - 1) / 2) * 0.018 * ctx.unit

                x0 = ax0 + base_offset * math.cos(perp)
                y0 = ay0 + base_offset * math.sin(perp)

                # Small lengths, like subtle fuzz
                L = (0.055 + 0.010 * i) * ctx.unit

                x1 = x0 + L * math.cos(ang)
                y1 = y0 + L * math.sin(ang)

                line, = ctx.ax.plot(
                    [x0, x1],
                    [y0, y1],
                    color=fuzz_color,
                    linewidth=1.0,
                    alpha=0.42,
                    zorder=fh_z,
                    solid_capstyle="round"
                )

        draw_facial_hair_curls(
            ctx,
            hair_color=hair_color,
            mouth_cx=mouth_cx,
            mouth_cy=mouth_cy,
            mouth_w=mouth_w,
            mouth_h=mouth_h,
            fh_type=fh_type,
            zorder=z + 3
        )

        return layout

    # --------------------------------------------------
    # 1) GOATEE — Homer-Simpson-like muzzle/chin beard
    # --------------------------------------------------
    elif fh_type == "goatee":
        # Put facial hair behind the mouth
        fh_z = z - 2

        # Build a filled "muzzle" around the mouth.
        # Top sits a little above mouth center so the mouth can cut across it.
        top_y = mouth_cy + 0.22 * mouth_h
        side_y = mouth_cy - 0.18 * mouth_h
        bottom_y = mouth_cy - 1.05 * mouth_h

        goatee_pts = [
            [mouth_cx - 0.55 * mouth_w, top_y],
            [mouth_cx + 0.55 * mouth_w, top_y],

            [mouth_cx + 0.68 * mouth_w, side_y],
            [mouth_cx + 0.40 * mouth_w, bottom_y],

            [mouth_cx,                 bottom_y - 0.18 * mouth_h],

            [mouth_cx - 0.40 * mouth_w, bottom_y],
            [mouth_cx - 0.68 * mouth_w, side_y],
        ]

        goatee = Polygon(
            goatee_pts,
            closed=True,
            facecolor=hair_color,
            edgecolor="black",
            linewidth=1.0,
            zorder=fh_z,
            joinstyle="round"
        )
        ctx.ax.add_patch(goatee)

        draw_facial_hair_curls(
            ctx,
            hair_color=hair_color,
            mouth_cx=mouth_cx,
            mouth_cy=mouth_cy,
            mouth_w=mouth_w,
            mouth_h=mouth_h,
            fh_type=fh_type,
            zorder=z + 3
        )

        return layout

    # --------------------------------------------------
    # 2) BEARD — Homer muzzle + draping lower beard
    # --------------------------------------------------
    elif fh_type == "beard":
        fh_z = z - 2

        # Upper part: same Homer-like face beard
        top_y = mouth_cy + 0.24 * mouth_h
        side_y = mouth_cy - 0.16 * mouth_h
        chin_y = mouth_cy - 0.95 * mouth_h

        # Lower drape extends off the face more like a beard wedge
        beard_tip_y = mouth_cy - 2.05 * mouth_h

        beard_pts = [
            [mouth_cx - 0.58 * mouth_w, top_y],
            [mouth_cx + 0.58 * mouth_w, top_y],

            [mouth_cx + 0.74 * mouth_w, side_y],
            [mouth_cx + 0.54 * mouth_w, chin_y],
            [mouth_cx + 0.24 * mouth_w, mouth_cy - 1.45 * mouth_h],

            [mouth_cx, beard_tip_y],

            [mouth_cx - 0.24 * mouth_w, mouth_cy - 1.45 * mouth_h],
            [mouth_cx - 0.54 * mouth_w, chin_y],
            [mouth_cx - 0.74 * mouth_w, side_y],
        ]

        beard = Polygon(
            beard_pts,
            closed=True,
            facecolor=hair_color,
            edgecolor="black",
            linewidth=1.0,
            zorder=fh_z,
            joinstyle="round"
        )
        ctx.ax.add_patch(beard)

        draw_facial_hair_curls(
            ctx,
            hair_color=hair_color,
            mouth_cx=mouth_cx,
            mouth_cy=mouth_cy,
            mouth_w=mouth_w,
            mouth_h=mouth_h,
            fh_type=fh_type,
            zorder=z + 3
        )

        return layout
    # --------------------------------------------------
    # 3) PEDO — two small marks
    # --------------------------------------------------
    elif fh_type == "pedo":
        y = layout["upper_band"]["y"]
        for side in [-1, 1]:
            dot = Ellipse(
                (mouth_cx + side * 0.22 * mouth_w, y),
                width=0.4 * mouth_w,
                height=0.25 * mouth_h,
                angle=side * -10,
                facecolor=hair_color,
                edgecolor="black",
                linewidth=0.8,
                zorder=z
            )
            ctx.ax.add_patch(dot)

        draw_facial_hair_curls(
            ctx,
            hair_color=hair_color,
            mouth_cx=mouth_cx,
            mouth_cy=mouth_cy,
            mouth_w=mouth_w,
            mouth_h=mouth_h,
            fh_type=fh_type,
            zorder=z + 3
        )

        return layout

    # --------------------------------------------------
    # 4) CURLY — two-lobed
    # --------------------------------------------------
    elif fh_type == "curly":
        y = layout["upper_band"]["y"]

        for side in [-1, 1]:
            x0 = mouth_cx + side * 0.05 * mouth_w
            x1 = mouth_cx + side * 0.58 * mouth_w

            verts = [
                (x0, y),
                (mouth_cx + side * 0.18 * mouth_w, y - 0.33 * mouth_h),
                (mouth_cx + side * 0.42 * mouth_w, y - 0.12 * mouth_h),
                (x1, y + 0.35 * mouth_h),
            ]

            patch = PathPatch(
                Path(
                    verts,
                    [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]
                ),
                facecolor="none",
                edgecolor=hair_color,
                linewidth=3.0,
                zorder=z,
                capstyle="round"
            )
            ctx.ax.add_patch(patch)

            """
            # little curl tip
            curl = Arc(
                (x1, y + 0.2 * mouth_h),
                width=0.18 * mouth_w,
                height=0.20 * mouth_h,
                theta1=210 if side == -1 else -30,
                theta2=360 if side == -1 else 120,
                color=hair_color,
                linewidth=2.4,
                zorder=z
            )
            ctx.ax.add_patch(curl)
            """

        draw_facial_hair_curls(
            ctx,
            hair_color=hair_color,
            mouth_cx=mouth_cx,
            mouth_cy=mouth_cy,
            mouth_w=mouth_w,
            mouth_h=mouth_h,
            fh_type=fh_type,
            zorder=z + 3
        )

        return layout

    # --------------------------------------------------
    # 5) CHAPMAN — little soul patch
    # --------------------------------------------------
    elif fh_type == "chapman":
        y = layout["upper_band"]["y"] - 0.05 * mouth_h

        patch = Polygon(
            [
                [mouth_cx - 0.18 * mouth_w, y + 0.05 * mouth_h],
                [mouth_cx + 0.18 * mouth_w, y + 0.05 * mouth_h],
                [mouth_cx + 0.22 * mouth_w, y - 0.18 * mouth_h],
                [mouth_cx - 0.22 * mouth_w, y - 0.18 * mouth_h],
            ],
            closed=True,
            facecolor="none",
            edgecolor=hair_color,
            linewidth=2.6,
            zorder=z,
            joinstyle="round"
        )
        ctx.ax.add_patch(patch)

        draw_facial_hair_curls(
            ctx,
            hair_color=hair_color,
            mouth_cx=mouth_cx,
            mouth_cy=mouth_cy,
            mouth_w=mouth_w,
            mouth_h=mouth_h,
            fh_type=fh_type,
            zorder=z + 3
        )

        return layout

    # --------------------------------------------------
    # 6) SOL — small chin tuft
    # --------------------------------------------------
    elif fh_type == "sol":
        y = layout["chin_band"]["y"]

        goatee = Polygon(
            [
                [mouth_cx - 0.20 * mouth_w, y + 0.20 * mouth_h],
                [mouth_cx + 0.20 * mouth_w, y + 0.20 * mouth_h],
                [mouth_cx + 0.12 * mouth_w, y - 0.44 * mouth_h],
                [mouth_cx,                 y - 0.68 * mouth_h],
                [mouth_cx - 0.12 * mouth_w, y - 0.44 * mouth_h],
            ],
            closed=True,
            facecolor=hair_color,
            edgecolor="black",
            linewidth=0.8,
            zorder=z,
            joinstyle="round"
        )
        ctx.ax.add_patch(goatee)

        draw_facial_hair_curls(
            ctx,
            hair_color=hair_color,
            mouth_cx=mouth_cx,
            mouth_cy=mouth_cy,
            mouth_w=mouth_w,
            mouth_h=mouth_h,
            fh_type=fh_type,
            zorder=z + 3
        )

        return layout

    return layout

def draw_rock(rock, ax=None, show_genes=False, normalize_size=True):
    """
    Trait-based rock renderer.

    Uses the new categorical/co-dominant visual phenotype.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 4))

    rng = np.random.default_rng(1_000_000 + rock.id)
    py_rng = random.Random(200_000 + rock.id)

    v = get_visual_phenotype(rock)

    shape_name = v.get("shape", "circle")
    size_name = v.get("size", "medium")
    color_name = v.get("color", "brown")
    color_alleles = v.get("color_alleles", [color_name])

    body_points, s = make_body_points(shape_name, size_name, rng)
    body_color = get_body_color_from_alleles(color_alleles)

    body = Polygon(
        body_points,
        closed=True,
        facecolor=body_color,
        edgecolor="black",
        linewidth=2.0,
        zorder=1
    )

    ctx = RockRenderContext(
    ax=ax,
    rock=rock,
    v=v,
    rng=rng,
    py_rng=py_rng,
    body=body,
    body_points=body_points,
    s=s,
    body_color=body_color
    )

    draw_wings(ctx)
    draw_fuzz(ctx)
    draw_halo(ctx)

    draw_stones(ctx)
    draw_tail(ctx)

    draw_horns(ctx)

    ax.add_patch(body)
    draw_patchwork(ax, body, color_alleles, s, rng)
    draw_hair(ctx, rock, v)

    draw_ears(ctx)

    draw_wrinkles(ctx)
    draw_freckles(ctx)

    draw_arms(ctx)
    draw_crown(ctx)

    drawn_eye_positions = draw_eyes(ctx)
    draw_brows(ctx, drawn_eye_positions)
    nose_info = draw_nose(ctx, drawn_eye_positions)
    mouth_info = draw_mouth(ctx, drawn_eye_positions)
    draw_facial_hair(
      ctx,
      rock,
      v,
      drawn_eye_positions=drawn_eye_positions,
      nose_info=nose_info,
      mouth_info=mouth_info
      )

    # Temporary compatibility for any older code below this point.
    eye_positions = [(x, y) for x, y, r in drawn_eye_positions]

    # -----------------------------
    # Craisen overlay
    # -----------------------------

    if v.get("is_craisen", False):
        ax.plot([-0.75 * s, 0.75 * s], [-0.75 * s, 0.75 * s], color="crimson", linewidth=4, zorder=20)
        ax.plot([-0.75 * s, 0.75 * s], [0.75 * s, -0.75 * s], color="crimson", linewidth=4, zorder=20)
        ax.text(0, -1.25 * s, "CRAISEN", color="crimson", ha="center", va="center", fontsize=10, fontweight="bold")

    # -----------------------------
    # Labels / formatting
    # -----------------------------

    ax.set_title(f"{rock.name} #{rock.id}\nGen {rock.generation}")
    ax.set_aspect("equal")

    if normalize_size:
        # Portrait mode:
        # each rock fills its own frame.
        ax.set_xlim(-2.0 * s, 2.0 * s)
        ax.set_ylim(-1.65 * s, 1.75 * s)
    else:
        # Comparison mode:
        # every rock uses the same camera.
        # This makes small/large/giant visibly different.
        ax.set_xlim(-3.0, 3.0)
        ax.set_ylim(-2.55, 2.85)

    ax.axis("off")

    if show_genes:
        gene_text = "\n".join([f"{k}: {v}" for k, v in rock.genes.items()])
        ax.text(
            1.55 * s,
            1.35 * s,
            gene_text,
            fontsize=7,
            va="top",
            family="monospace"
        )

    return ax

def show_rocks(
    rock_items,
    rock_source=None,
    cols=6,
    figsize_per_rock=3.2,
    show_genes=False,
    show_traits=False,
    title=None,
    sort_by_generation=False,
    normalize_size=True
):
    """
    Display a grid of rocks.

    Accepts:
    - a dictionary of rocks: show_rocks(rocks)
    - a list of rock IDs: show_rocks([1, 2, 3], rock_source=rocks)
    - a list of Rock objects: show_rocks([rock1, rock2])
    - a dictionary of test rocks: show_rocks(test_rocks)

    Parameters
    ----------
    rock_items:
        Dict[int, Rock], list[int], tuple[int], list[Rock], or tuple[Rock]

    rock_source:
        Optional dictionary used when rock_items is a list of IDs.
        If None, the function tries to use the global `rocks`.

    cols:
        Number of columns in the display grid.

    figsize_per_rock:
        Size multiplier for each rock subplot.

    show_genes:
        Passes show_genes=True into draw_rock.

    show_traits:
        Adds a compact trait label under each rock.

    title:
        Optional figure title.

    sort_by_generation:
        If True, sorts rocks by generation, then ID.
    """

    # -----------------------------
    # Resolve input into Rock objects
    # -----------------------------

    if isinstance(rock_items, dict):
        rock_list = list(rock_items.values())

    else:
        rock_list = []

        for item in list(rock_items):
            if isinstance(item, Rock):
                rock_list.append(item)

            elif isinstance(item, int):
                source = rock_source

                if source is None:
                    try:
                        source = rocks
                    except NameError:
                        raise ValueError(
                            "You passed rock IDs, but no rock_source was provided "
                            "and no global `rocks` dictionary exists."
                        )

                if item not in source:
                    raise KeyError(f"Rock ID {item} was not found in the provided rock source.")

                rock_list.append(source[item])

            else:
                raise TypeError(
                    "show_rocks expects a dict of rocks, a list of Rock objects, "
                    "or a list of integer rock IDs."
                )

    if sort_by_generation:
        rock_list = sorted(rock_list, key=lambda r: (r.generation, r.id))

    n = len(rock_list)

    if n == 0:
        print("No rocks to show.")
        return None, None

    # -----------------------------
    # Create grid
    # -----------------------------

    cols = max(1, min(cols, n))
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(cols * figsize_per_rock, rows * figsize_per_rock)
    )

    axes = np.array(axes).reshape(-1)

    # Turn all axes off first.
    for ax in axes:
        ax.axis("off")

    # -----------------------------
    # Draw rocks
    # -----------------------------

    for ax, rock in zip(axes, rock_list):
        draw_rock(rock, ax=ax, show_genes=show_genes, normalize_size=normalize_size)

        pad_rock_axis(ax, pad_frac=PAD_FRAC)

        if show_traits:
            v = get_visual_phenotype(rock)

            trait_text = (
                f"{v.get('shape', 'n/a')} | {v.get('size', 'n/a')} | {v.get('color', 'n/a')}\n"
                f"eyes: {v.get('eyes', 'n/a')} | hair: {v.get('hair', 'n/a')} | {v.get('hair_color', 'n/a')}"
            )

            if v.get("is_craisen", False):
                trait_text += "\nCRAISEN"

            ax.text(
                0.5,
                -0.08,
                trait_text,
                transform=ax.transAxes,
                ha="center",
                va="top",
                fontsize=8
            )

    if title is not None:
        fig.suptitle(title, fontsize=16, y=1.02)

    plt.tight_layout()
    plt.show()

    return fig, axes

def rock_to_image_uri(rock, sprite_size=2.0, dpi=400):
    """
    Render a rock to a transparent PNG and return it as a base64 image URI
    that Plotly can place on the graph.
    """
    fig, ax = plt.subplots(figsize=(sprite_size, sprite_size), dpi=dpi)

    draw_rock(rock, ax=ax)

    pad_rock_axis(ax, pad_frac=PAD_FRAC)

    # Remove the title from the mini image.
    ax.set_title("")
    ax.axis("off")

    fig.patch.set_alpha(0)
    ax.set_facecolor((0, 0, 0, 0))

    buffer = io.BytesIO()
    fig.savefig(
        buffer,
        format="png",
        transparent=True,
        bbox_inches="tight",
        pad_inches=0.1,
        dpi = dpi
    )
    plt.close(fig)

    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return "data:image/png;base64," + encoded

def pad_rock_axis(ax, pad_frac=PAD_FRAC):
    """
    Expand the current axes limits so external traits do not get clipped.

    Good for halos, ion stones, wings, tails, hair, horns, etc.
    """
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()

    dx = x1 - x0
    dy = y1 - y0

    ax.set_xlim(x0 - pad_frac * dx, x1 + pad_frac * dx)
    ax.set_ylim(y0 - pad_frac * dy, y1 + pad_frac * dy)
    ax.set_aspect("equal")

def compute_lineage_positions_christmas(
    rocks,
    x_gap=5,
    gen_gap=5,
    parent_pair_gap=3,
    min_node_gap=3,
    anti_overlap=True,
    layout_passes=4
):
    """
    Robust Christmas-tree lineage layout.

    Goals:
    - all rocks get positions
    - parents are pulled toward their children
    - missing/orphan parent links do not crash the tree
    - sold/dead/puffed/spore clones can remain visible
    - works even when weird generation relationships appear

    Returns:
        pos = {rock_id: (x, y)}
    """

    if rocks is None or len(rocks) == 0:
        return {}

    # Make sure IDs are normal ints when possible.
    rock_ids = list(rocks.keys())

    # --------------------------------------------------
    # Group rocks by generation
    # --------------------------------------------------
    by_generation = {}

    for rid, rock in rocks.items():
        gen = getattr(rock, "generation", 0)

        if gen is None:
            gen = 0

        by_generation.setdefault(gen, []).append(rid)

    generations = sorted(by_generation.keys())

    # --------------------------------------------------
    # Initial centered layout within each generation
    # --------------------------------------------------
    pos_x = {}

    for gen in generations:
        ids = sorted(by_generation[gen])
        n = len(ids)

        if n == 1:
            pos_x[ids[0]] = 0.0
        else:
            start_x = -0.5 * (n - 1) * x_gap

            for i, rid in enumerate(ids):
                pos_x[rid] = start_x + i * x_gap

    # Ensure every rock has an x position.
    for rid in rock_ids:
        if rid not in pos_x:
            pos_x[rid] = 0.0

    # --------------------------------------------------
    # Pull parents toward their children, bottom-up
    # --------------------------------------------------
    for _ in range(layout_passes):
        desired_x = {rid: [] for rid in rock_ids}

        # For each child, request parent positions around child center.
        for child_id, child in rocks.items():
            parents = getattr(child, "parents", None)

            if parents is None:
                continue

            if len(parents) != 2:
                continue

            p1, p2 = parents

            # Defensive skip: parent might not exist in current tree dictionary.
            if p1 not in rocks or p2 not in rocks:
                continue

            if child_id not in pos_x:
                continue

            child_center_x = pos_x[child_id]

            # Make sure keys exist even if the old data is odd.
            desired_x.setdefault(p1, [])
            desired_x.setdefault(p2, [])

            desired_x[p1].append(child_center_x - parent_pair_gap / 2)
            desired_x[p2].append(child_center_x + parent_pair_gap / 2)

        # Update positions from desired child-centered positions.
        # Work from older generations first to keep the tree stable.
        for gen in generations:
            for rid in by_generation[gen]:
                if rid in desired_x and len(desired_x[rid]) > 0:
                    old_x = pos_x.get(rid, 0.0)
                    target_x = sum(desired_x[rid]) / len(desired_x[rid])

                    # Blend instead of snapping to reduce wild oscillations.
                    pos_x[rid] = 0.45 * old_x + 0.55 * target_x

        # --------------------------------------------------
        # Anti-overlap pass within each generation
        # --------------------------------------------------
        if anti_overlap:
            for gen in generations:
                ids = sorted(by_generation[gen], key=lambda rid: pos_x.get(rid, 0.0))

                if len(ids) <= 1:
                    continue

                # Left-to-right push
                for i in range(1, len(ids)):
                    prev_id = ids[i - 1]
                    curr_id = ids[i]

                    if pos_x[curr_id] - pos_x[prev_id] < min_node_gap:
                        pos_x[curr_id] = pos_x[prev_id] + min_node_gap

                # Recenter the generation around zero-ish
                mean_x = sum(pos_x[rid] for rid in ids) / len(ids)

                for rid in ids:
                    pos_x[rid] -= mean_x

    # --------------------------------------------------
    # Final positions
    # --------------------------------------------------
    pos = {}

    for rid, rock in rocks.items():
        gen = getattr(rock, "generation", 0)

        if gen is None:
            gen = 0

        x = pos_x.get(rid, 0.0)
        y = -gen * gen_gap

        pos[rid] = (x, y)

    return pos

FAMILY_PALETTE = (
    px.colors.qualitative.Safe
    + px.colors.qualitative.Set2
    + px.colors.qualitative.Pastel
    + px.colors.qualitative.Bold
)

def family_color(parent_pair):
    """
    Deterministic color for a parent pair.
    Same parent pair -> same color every time.
    """
    key = f"{min(parent_pair)}-{max(parent_pair)}"
    h = int(hashlib.md5(key.encode()).hexdigest(), 16)
    return FAMILY_PALETTE[h % len(FAMILY_PALETTE)]

FAMILY_COLORS = [
    "#4E79A7",  # blue
    "#F28E2B",  # orange
    "#59A14F",  # green
    "#E15759",  # red
    "#B07AA1",  # purple
    "#76B7B2",  # teal
    "#EDC948",  # yellow
    "#9C755F",  # brown
    "#FF9DA7",  # pink
    "#BAB0AC",  # gray
]

FAMILY_DASHES = [
    "solid",
    "dash",
    "dot",
    "longdash",
    "dashdot",
]

def get_family_styles(rocks):
    """
    Assign a stable color/dash style to each parent pair.
    """
    families = []

    for child_id, rock in rocks.items():
        if rock.parents is not None:
            families.append(tuple(sorted(rock.parents)))

    families = sorted(set(families))

    style_map = {}

    for i, fam in enumerate(families):
        style_map[fam] = {
            "color": FAMILY_COLORS[i % len(FAMILY_COLORS)],
            "dash": FAMILY_DASHES[(i // len(FAMILY_COLORS)) % len(FAMILY_DASHES)]
        }

    return style_map

def build_family_segments(pos, parent_pair, child_ids):
    """
    Builds pedigree line segments for one parent pair and its displayed children.
    """
    p1, p2 = parent_pair

    if p1 not in pos or p2 not in pos:
        return [], []

    child_ids = [cid for cid in child_ids if cid in pos]

    if len(child_ids) == 0:
        return [], []

    child_ids = sorted(child_ids, key=lambda cid: pos[cid][0])

    x1, y1 = pos[p1]
    x2, y2 = pos[p2]

    child_xs = [pos[cid][0] for cid in child_ids]
    child_ys = [pos[cid][1] for cid in child_ids]

    child_y = child_ys[0]
    parent_y = min(y1, y2)

    parent_bar_y = parent_y - 0.48
    sibling_bar_y = child_y + 0.68

    parent_center_x = (x1 + x2) / 2
    child_center_x = sum(child_xs) / len(child_xs)

    line_segments_x = []
    line_segments_y = []

    def add_segment(xa, ya, xb, yb):
        line_segments_x.extend([xa, xb, None])
        line_segments_y.extend([ya, yb, None])

    # Parent drops.
    add_segment(x1, y1 - 0.45, x1, parent_bar_y)
    add_segment(x2, y2 - 0.45, x2, parent_bar_y)

    # Parent pair bar.
    add_segment(x1, parent_bar_y, x2, parent_bar_y)

    # Descent toward children.
    add_segment(parent_center_x, parent_bar_y, child_center_x, sibling_bar_y)

    # Sibling bar and child drops.
    if len(child_ids) > 1:
        add_segment(min(child_xs), sibling_bar_y, max(child_xs), sibling_bar_y)

        for cx in child_xs:
            add_segment(cx, sibling_bar_y, cx, child_y + 0.45)
    else:
        cx = child_xs[0]
        add_segment(child_center_x, sibling_bar_y, cx, child_y + 0.45)

    return line_segments_x, line_segments_y

SIZE_SCALE_MAP = {
    "medium": 1.00,
    "large": 1.22,
    "small": 0.78,
    "giant": 1.55,
    "missized": 1.10,
}

def get_rock_size_scale(rock):
    """
    Returns the expressed visual size scale for a rock.
    """
    v = get_visual_phenotype(rock)
    return SIZE_SCALE_MAP.get(v.get("size", "medium"), 1.0)

def get_gender_symbol(rock):
    """
    Return display symbol for rock gender.
    """
    gender = get_rock_gender_value(rock)

    if gender == 1:
        return "♂"

    return "♀"

def get_gender_color(rock):
    """
    Display color for gender symbol.
    """
    gender = get_rock_gender_value(rock)

    if gender == 1:
        return "royalblue"

    return "deeppink"

def clean_hover_value(value):
    """
    Make phenotype values readable in Plotly hover text.
    """
    if isinstance(value, float):
        return f"{value:.3g}"

    if isinstance(value, (list, tuple)):
        return ", ".join(str(v) for v in value)

    if isinstance(value, dict):
        return ", ".join(f"{k}: {v}" for k, v in value.items())

    return str(value)

def format_full_phenotype_hover(rock):
    """
    Build a full phenotype printout for hover boxes.
    """
    v = get_visual_phenotype(rock)

    lines = []

    for key in sorted(v.keys()):
        value = clean_hover_value(v[key])
        lines.append(f"{key}: {value}")

    return "<br>".join(lines)

def is_rock_sold_flag(rock):
    return bool(getattr(rock, "sold", False))

def get_rock_status_symbol(rock):
    """
    Symbol shown near rocks in game views.
    """
    if getattr(rock, "puffed", False):
        return "☁"

    if getattr(rock, "dead", False):
        return "†"

    if is_rock_sold_flag(rock):
        return "$"

    if getattr(rock, "is_craisen", 0) == 1:
        return "X"

    if getattr(rock, "used_as_parent", False):
        return "○"

    return ""

def get_rock_status_color(rock):
    """
    Color for status symbols.
    """
    if getattr(rock, "puffed", False):
        return "dimgray"

    if getattr(rock, "dead", False):
        return "black"

    if is_rock_sold_flag(rock):
        return "green"

    if getattr(rock, "is_craisen", 0) == 1:
        return "crimson"

    if getattr(rock, "used_as_parent", False):
        return "gray"

    return "black"

def get_pair_mutation_rate(base_mutation_rate, potion_key):
    """
    Mutation potion increases mutation rate for this pair.
    """
    potion_key = normalize_potion_key(potion_key)

    if potion_key == "mutation":
        return POTION_MUTATION_RATE

    return base_mutation_rate

def apply_fertility_to_clutch(clutch_size, potion_key):
    """
    Fertility potion adds extra children to the clutch.
    """
    potion_key = normalize_potion_key(potion_key)

    if potion_key == "fertility":
        return clutch_size + FERTILITY_EXTRA_CHILDREN

    return clutch_size

def maybe_kill_child(child, death_chance=CHILD_DEATH_CHANCE):
    """
    Child has a percent chance of dying after birth.

    Dead children remain in the tree but are worthless and cannot breed.
    """
    ensure_rock_game_attributes(child)

    if random.random() < death_chance:
        child.dead = True
        child.death_reason = "died after birth"
        child.sell_value = 0
        child.score_value = 0
        return True

    return False

def roll_clutch_size(mean=CLUTCH_MEAN, std=CLUTCH_STD, max_clutch_size=MAX_CLUTCH_SIZE):
    """
    Emulates Excel:
    ABS(INT(NORMINV(RAND(), 1.5, 2))) + 1

    In Python:
    NORMINV(RAND(), mean, std) is equivalent to a normal draw.
    Excel INT floors toward negative infinity, so use math.floor.
    """
    x = random.gauss(mean, std)
    clutch = abs(math.floor(x)) + 1

    if max_clutch_size is not None:
        clutch = min(clutch, max_clutch_size)

    return clutch

def run_generation_from_ui(game):
    """
    UI-safe generation executor.

    Uses the newer clutch/death/spore generation system.
    """
    return execute_breeding_generation(
        game,
        mutation_rate=0.02,
        child_death_chance=CHILD_DEATH_CHANCE,
        use_clutch_formula=True,
        fixed_children_per_pair=1,
        handle_mitosion=True,
        handle_sporing=True,
        abort_on_invalid=True,
        show_results=True
    )

def apply_reroll_to_clutch(clutch_size, potion_key):
    """
    Take the best of two rolls for the clutch.
    """
    potion_key = normalize_potion_key(potion_key)

    if potion_key == "reroll":
        return max(clutch_size, roll_clutch_size())

    return clutch_size

def get_rock(game, rock_id):
    """
    Safely get a rock by id.

    Accepts:
    - integer id
    - numeric string id
    - Rock object
    """
    if rock_id is None:
        return None

    if hasattr(rock_id, "id"):
        rock_id = rock_id.id

    try:
        rock_id = int(rock_id)
    except Exception:
        return None

    return game.rocks.get(rock_id, None)

def get_parent_ids(rock):
    """
    Return parent IDs as a set.
    Founders/imports have no parents.
    """
    if rock.parents is None:
        return set()

    return set(rock.parents)

def are_siblings(rock_a, rock_b):
    """
    Two rocks are siblings if they share at least one parent.

    Founders/imports with parents=None are not siblings.
    """
    parents_a = get_parent_ids(rock_a)
    parents_b = get_parent_ids(rock_b)

    if len(parents_a) == 0 or len(parents_b) == 0:
        return False

    return len(parents_a & parents_b) > 0

def is_parent_child(rock_a, rock_b):
    """
    True if one rock is a direct parent of the other.
    """
    parents_a = get_parent_ids(rock_a)
    parents_b = get_parent_ids(rock_b)

    return (rock_a.id in parents_b) or (rock_b.id in parents_a)

def split_gene_alleles(gene_name, gene_value):
    """
    Split a gene pair into two alleles.

    Normal categorical genes:
        "34" -> ["3", "4"]

    Death genes:
        "0517" -> ["05", "17"]

    Gender:
        "01" -> ["0", "1"]
    """
    raw = str(gene_value)

    if gene_name in DEATH_GENES:
        raw = raw.zfill(4)
        return [raw[:2], raw[2:4]]

    # Normal genes are two single-character categorical alleles.
    if len(raw) == 1:
        raw = raw + raw

    return [raw[0], raw[1]]

def join_gene_alleles(gene_name, allele_a, allele_b):
    """
    Recombine two inherited alleles into stored gene format.

    Gender is normalized so:
    10 -> 01
    """
    if gene_name in DEATH_GENES:
        return f"{int(allele_a):02d}{int(allele_b):02d}"

    if gene_name == "gender":
        alleles = sorted([str(allele_a), str(allele_b)])
        return "".join(alleles)

    return f"{allele_a}{allele_b}"

def random_valid_allele_for_gene(gene_name):
    """
    Used for mutation.

    Chooses a valid allele for the given gene.
    """
    if gene_name in DEATH_GENES:
        return f"{random.randint(1, 99):02d}"

    if gene_name == "gender":
        return random.choice(["0", "1"])

    if gene_name in Rock_roll_dict:
        valid_values = Rock_roll_dict[gene_name][1]
        return str(random.choice(valid_values))

    # Fallback for unknown two-state genes.
    return random.choice(["0", "1"])

def maybe_mutate_allele(gene_name, allele, mutation_rate=0.02):
    """
    Mutate an allele with probability mutation_rate.
    """
    if random.random() < mutation_rate:
        return random_valid_allele_for_gene(gene_name)

    return allele

def inherit_gene_from_parents(gene_name, parent_a_gene, parent_b_gene, mutation_rate=0.02):
    """
    Child gets one allele from each parent.
    """
    alleles_a = split_gene_alleles(gene_name, parent_a_gene)
    alleles_b = split_gene_alleles(gene_name, parent_b_gene)

    allele_a = random.choice(alleles_a)
    allele_b = random.choice(alleles_b)

    allele_a = maybe_mutate_allele(gene_name, allele_a, mutation_rate)
    allele_b = maybe_mutate_allele(gene_name, allele_b, mutation_rate)

    return join_gene_alleles(gene_name, allele_a, allele_b)

def get_ancestors(game, rock_id, include_self=False):
    """
    Return all ancestors of a rock.
    """
    rock = get_rock(game, rock_id)

    if rock is None:
        return set()

    ancestors = set()

    if include_self:
        ancestors.add(rock_id)

    def walk(r):
        for parent_id in get_parent_ids(r):
            if parent_id not in ancestors:
                ancestors.add(parent_id)
                parent = get_rock(game, parent_id)
                if parent is not None:
                    walk(parent)

    walk(rock)

    return ancestors

def get_rock_gender_value(rock):
    """
    Return gender as:
    1 = Male
    0 = Female

    Handles both stored fields and gene fallback.
    """
    try:
        v = get_visual_phenotype(rock)
        g = v.get("gender", None)

        if isinstance(g, str):
            return 1 if g.lower() == "male" else 0

        if g in [0, 1]:
            return int(g)
    except Exception:
        pass

    if hasattr(rock, "gender") and rock.gender in [0, 1]:
        return int(rock.gender)

    gender_gene = str(rock.genes.get("gender", "00"))

    return 1 if gender_gene == "01" else 0

def get_rock_gender_name(rock):
    return "Male" if get_rock_gender_value(rock) == 1 else "Female"

def is_rock_sold(rock):
    return bool(getattr(rock, "sold", False))

def is_rock_craisen(rock):
    """
    Refreshes craisen status if possible.
    """
    try:
        evaluate_rock_value(rock)
    except Exception:
        pass

    return bool(getattr(rock, "is_craisen", 0) == 1)


# ============================================================
# RELATEDNESS / INBREEDING HELPERS
# ============================================================

DISTANT_RELATIONSHIP_R_THRESHOLD = 1.0 / 32.0


def _safe_int_id(value):
    """
    Convert rock id-like values into int ids.

    Supports:
    - int
    - numeric string
    - Rock object with .id
    """
    if value is None:
        return None

    if hasattr(value, "id"):
        value = getattr(value, "id")

    try:
        return int(value)
    except Exception:
        return None


def get_rock_by_id_safe(game, rock_id):
    """
    Safely fetch a rock from game.rocks.
    """
    rock_id = _safe_int_id(rock_id)

    if rock_id is None:
        return None

    if game is None or not hasattr(game, "rocks"):
        return None

    if rock_id in game.rocks:
        return game.rocks[rock_id]

    if str(rock_id) in game.rocks:
        return game.rocks[str(rock_id)]

    return None


def get_parent_ids_for_relationship(game, rock_id):
    """
    Robustly find parent ids for a rock.

    This supports several possible data models:
    - rock.parents
    - rock.parent_ids
    - rock.parent_pair
    - rock.parent_a_id / rock.parent_b_id
    - rock.mother_id / rock.father_id
    """
    rock = get_rock_by_id_safe(game, rock_id)

    if rock is None:
        return []

    # Common single attributes that may hold tuple/list/set/dict.
    container_attrs = [
        "parent_ids",
        "parents",
        "parent_pair",
        "parent_id_pair",
    ]

    for attr in container_attrs:
        if not hasattr(rock, attr):
            continue

        value = getattr(rock, attr)

        if callable(value):
            value = value()

        if value is None:
            continue

        if isinstance(value, dict):
            raw_values = list(value.values())
        elif isinstance(value, (list, tuple, set)):
            raw_values = list(value)
        else:
            raw_values = [value]

        cleaned = []

        for raw in raw_values:
            parent_id = _safe_int_id(raw)
            if parent_id is not None:
                cleaned.append(parent_id)

        if len(cleaned) > 0:
            return list(dict.fromkeys(cleaned))

    # Common paired attributes.
    paired_attrs = [
        ("parent_a_id", "parent_b_id"),
        ("parent1_id", "parent2_id"),
        ("mother_id", "father_id"),
        ("mom_id", "dad_id"),
        ("parent_a", "parent_b"),
        ("mother", "father"),
    ]

    cleaned = []

    for attr_a, attr_b in paired_attrs:
        for attr in [attr_a, attr_b]:
            if not hasattr(rock, attr):
                continue

            parent_id = _safe_int_id(getattr(rock, attr))

            if parent_id is not None:
                cleaned.append(parent_id)

        if len(cleaned) > 0:
            return list(dict.fromkeys(cleaned))

    return []


def get_ancestor_path_distances(game, rock_id, include_self=True, max_depth=12):
    """
    Return a dictionary:

        ancestor_id -> [distance_1, distance_2, ...]

    Distance:
        self = 0
        parent = 1
        grandparent = 2
        great-grandparent = 3

    Multiple distances are kept because inbred pedigrees may have multiple
    paths to the same ancestor.
    """
    rock_id = _safe_int_id(rock_id)

    if rock_id is None:
        return {}

    distances = {}

    stack = [
        (rock_id, 0, {rock_id})
    ]

    while len(stack) > 0:
        current_id, depth, path_seen = stack.pop()

        if depth > max_depth:
            continue

        if include_self or depth > 0:
            distances.setdefault(current_id, []).append(depth)

        if depth == max_depth:
            continue

        parent_ids = get_parent_ids_for_relationship(game, current_id)

        for parent_id in parent_ids:
            parent_id = _safe_int_id(parent_id)

            if parent_id is None:
                continue

            # Prevent accidental loops from bad saved data.
            if parent_id in path_seen:
                continue

            next_seen = set(path_seen)
            next_seen.add(parent_id)

            stack.append(
                (parent_id, depth + 1, next_seen)
            )

    return distances


def calculate_relatedness_r(game, rock_a_id, rock_b_id, max_depth=12):
    """
    Calculate coefficient of relationship R between two rocks.

    Formula:
        R = sum over shared ancestors of (1/2)^(distance_a + distance_b)

    Examples:
        parent-child: 1/2
        full siblings: 1/2
        half siblings: 1/4
        uncle/niece: 1/4
        first cousins: 1/8

    Returns:
        r, contributions

    contributions:
        dict ancestor_id -> contribution amount
    """
    rock_a_id = _safe_int_id(rock_a_id)
    rock_b_id = _safe_int_id(rock_b_id)

    if rock_a_id is None or rock_b_id is None:
        return 0.0, {}

    if rock_a_id == rock_b_id:
        return 1.0, {rock_a_id: 1.0}

    ancestors_a = get_ancestor_path_distances(
        game,
        rock_a_id,
        include_self=True,
        max_depth=max_depth
    )

    ancestors_b = get_ancestor_path_distances(
        game,
        rock_b_id,
        include_self=True,
        max_depth=max_depth
    )

    shared_ancestors = set(ancestors_a.keys()) & set(ancestors_b.keys())

    r = 0.0
    contributions = {}

    for ancestor_id in shared_ancestors:
        subtotal = 0.0

        for distance_a in ancestors_a[ancestor_id]:
            for distance_b in ancestors_b[ancestor_id]:
                subtotal += 0.5 ** (distance_a + distance_b)

        if subtotal > 0:
            contributions[ancestor_id] = subtotal
            r += subtotal

    # Keep weird pedigrees from producing silly display values.
    r = min(r, 1.0)

    return r, contributions


def ordinal_word(n):
    words = {
        1: "first",
        2: "second",
        3: "third",
        4: "fourth",
        5: "fifth",
        6: "sixth",
        7: "seventh",
        8: "eighth",
    }

    return words.get(int(n), f"{n}th")


def removal_word(n):
    words = {
        1: "once removed",
        2: "twice removed",
        3: "three times removed",
        4: "four times removed",
    }

    return words.get(int(n), f"{n} times removed")


def great_prefix(count):
    """
    count = 0 -> ''
    count = 1 -> 'great-'
    count = 2 -> '2x-great-'
    """
    count = int(count)

    if count <= 0:
        return ""

    if count == 1:
        return "great-"

    return f"{count}x-great-"


def describe_equivalent_relationship_from_r(r):
    """
    Describe approximate relationship class from R.
    """
    if r <= 0:
        return "unrelated / no known shared ancestor"

    if r >= 0.49:
        return "parent-child / full-sibling level"

    if r >= 0.249:
        return "half-sibling / aunt-uncle-niece-nephew / grandparent-grandchild level"

    if r >= 0.124:
        return "first-cousin level"

    if r >= 0.061:
        return "first-cousin-once-removed / great-aunt-uncle level"

    if r >= 0.031:
        return "second-cousin level"

    return "distant relation"


def describe_actual_relationship(game, rock_a_id, rock_b_id, max_depth=12):
    """
    Describe the closest actual relationship pattern between two rocks.
    """
    rock_a_id = _safe_int_id(rock_a_id)
    rock_b_id = _safe_int_id(rock_b_id)

    if rock_a_id is None or rock_b_id is None:
        return "unknown"

    if rock_a_id == rock_b_id:
        return "same rock"

    ancestors_a = get_ancestor_path_distances(
        game,
        rock_a_id,
        include_self=True,
        max_depth=max_depth
    )

    ancestors_b = get_ancestor_path_distances(
        game,
        rock_b_id,
        include_self=True,
        max_depth=max_depth
    )

    # Direct ancestor cases.
    if rock_a_id in ancestors_b:
        distance = min(ancestors_b[rock_a_id])

        if distance == 1:
            return "parent and child"

        if distance == 2:
            return "grandparent and grandchild"

        return f"{great_prefix(distance - 2)}grandparent and {great_prefix(distance - 2)}grandchild"

    if rock_b_id in ancestors_a:
        distance = min(ancestors_a[rock_b_id])

        if distance == 1:
            return "parent and child"

        if distance == 2:
            return "grandparent and grandchild"

        return f"{great_prefix(distance - 2)}grandparent and {great_prefix(distance - 2)}grandchild"

    shared_ancestors = set(ancestors_a.keys()) & set(ancestors_b.keys())

    if len(shared_ancestors) == 0:
        return "unrelated / no known shared ancestor"

    # Find closest shared-ancestor path pair.
    best = None

    for ancestor_id in shared_ancestors:
        for distance_a in ancestors_a[ancestor_id]:
            for distance_b in ancestors_b[ancestor_id]:
                if distance_a == 0 or distance_b == 0:
                    continue

                score = distance_a + distance_b

                if best is None or score < best[0]:
                    best = (score, distance_a, distance_b, ancestor_id)

    if best is None:
        return "unknown relation"

    _, distance_a, distance_b, _ = best

    min_d = min(distance_a, distance_b)
    max_d = max(distance_a, distance_b)

    # Siblings.
    if distance_a == 1 and distance_b == 1:
        common_parent_count = 0

        for ancestor_id in shared_ancestors:
            if 1 in ancestors_a[ancestor_id] and 1 in ancestors_b[ancestor_id]:
                common_parent_count += 1

        if common_parent_count >= 2:
            return "full siblings"

        return "half siblings"

    # Aunt/uncle style relation.
    if min_d == 1:
        great_count = max_d - 2

        if great_count <= 0:
            return "aunt/uncle and niece/nephew"

        return (
            f"{great_prefix(great_count)}aunt/uncle and "
            f"{great_prefix(great_count)}niece/nephew"
        )

    # Cousin relation.
    cousin_degree = min_d - 1
    removals = abs(distance_a - distance_b)

    cousin_text = f"{ordinal_word(cousin_degree)} cousins"

    if removals > 0:
        cousin_text += f" {removal_word(removals)}"

    return cousin_text

def format_relatedness_report(game, rock_a_id, rock_b_id, max_depth=12):
    """
    Human-readable relatedness report for breeding preview.
    """
    r, contributions = calculate_relatedness_r(
        game,
        rock_a_id,
        rock_b_id,
        max_depth=max_depth
    )

    f_child = r / 2.0

    equivalent = describe_equivalent_relationship_from_r(r)
    actual = describe_actual_relationship(
        game,
        rock_a_id,
        rock_b_id,
        max_depth=max_depth
    )

    if r > 0 and r < DISTANT_RELATIONSHIP_R_THRESHOLD:
        actual = "distant"

    return (
        f"R = {r:.4f} | estimated child F = {f_child:.4f}\n"
        f"Equivalent: {equivalent}\n"
        f"Actual: {actual}"
    )





def is_rock_breedable(rock):
    """
    Basic single-rock breedability.
    """
    if rock is None:
        return False, "Rock does not exist."

    ensure_rock_game_attributes(rock)

    if getattr(rock, "puffed", False):
        return False, f"Rock #{rock.id} puffed out and cannot breed."

    if getattr(rock, "dead", False):
        return False, f"Rock #{rock.id} is dead and cannot breed."

    if is_rock_sold(rock):
        return False, f"Rock #{rock.id} has been sold."

    if is_rock_craisen(rock):
        return False, f"Rock #{rock.id} is craisen and cannot breed."

    return True, "Breedable."

def game_get_parent_ids(rock):
    """
    Return parent IDs as a set.
    Founders/imports have no parents.
    """
    if rock is None:
        return set()

    if rock.parents is None:
        return set()

    return set(rock.parents)

def game_get_ancestors(game, rock_id, include_self=False):
    """
    Return all ancestor IDs of a rock using GameState.

    This intentionally avoids the old notebook function name get_ancestors().
    """
    rock_id = int(rock_id)

    if rock_id not in game.rocks:
        return set()

    found = set()

    if include_self:
        found.add(rock_id)

    def walk(current_id):
        if current_id not in game.rocks:
            return

        current_rock = game.rocks[current_id]

        for parent_id in game_get_parent_ids(current_rock):
            parent_id = int(parent_id)

            if parent_id not in found:
                found.add(parent_id)
                walk(parent_id)

    walk(rock_id)

    return found

def game_are_siblings(rock_a, rock_b):
    """
    Two rocks are siblings if they share at least one parent.
    Founders/imports with no parents are not siblings.
    """
    parents_a = game_get_parent_ids(rock_a)
    parents_b = game_get_parent_ids(rock_b)

    if len(parents_a) == 0 or len(parents_b) == 0:
        return False

    return len(parents_a & parents_b) > 0

def game_is_parent_child(rock_a, rock_b):
    """
    True if one rock is a direct parent of the other.
    """
    parents_a = game_get_parent_ids(rock_a)
    parents_b = game_get_parent_ids(rock_b)

    return (rock_a.id in parents_b) or (rock_b.id in parents_a)

def game_are_related(game, rock_a, rock_b):
    """
    Broad relationship check:
    siblings, parent-child, or shared ancestor.
    """
    if rock_a is None or rock_b is None:
        return False

    if game_are_siblings(rock_a, rock_b):
        return True

    if game_is_parent_child(rock_a, rock_b):
        return True

    ancestors_a = game_get_ancestors(game, rock_a.id)
    ancestors_b = game_get_ancestors(game, rock_b.id)

    if rock_a.id in ancestors_b or rock_b.id in ancestors_a:
        return True

    return len(ancestors_a & ancestors_b) > 0

def validate_breeding_pair(
    game,
    parent_a_id,
    parent_b_id,
    block_siblings=False,
    block_parent_child=False,
    require_opposite_gender=True,
    warn_related=True
):
    """
    Validate whether two rocks can breed.

    Uses game-specific relationship helpers so it cannot accidentally call
    the old get_ancestors(rocks, rock_id) function.
    """

    errors = []
    warnings = []

    try:
        parent_a_id = int(parent_a_id)
        parent_b_id = int(parent_b_id)
    except Exception:
        return {
            "valid": False,
            "errors": ["Parent IDs must be integers."],
            "warnings": [],
            "parent_a": None,
            "parent_b": None,
        }

    parent_a = get_rock(game, parent_a_id)
    parent_b = get_rock(game, parent_b_id)

    if parent_a is None:
        errors.append(f"Rock #{parent_a_id} does not exist.")

    if parent_b is None:
        errors.append(f"Rock #{parent_b_id} does not exist.")

    if parent_a is None or parent_b is None:
        return {
            "valid": False,
            "errors": errors,
            "warnings": warnings,
            "parent_a": parent_a,
            "parent_b": parent_b,
        }

    if parent_a_id == parent_b_id:
        errors.append("A rock cannot breed with itself.")

    ok_a, msg_a = is_rock_breedable(parent_a)
    ok_b, msg_b = is_rock_breedable(parent_b)

    if not ok_a:
        errors.append(msg_a)

    if not ok_b:
        errors.append(msg_b)

    gender_a = get_rock_gender_value(parent_a)
    gender_b = get_rock_gender_value(parent_b)

    if require_opposite_gender and gender_a == gender_b:
        errors.append(
            f"Parents must be opposite gender. "
            f"Rock #{parent_a.id} is {get_rock_gender_name(parent_a)} and "
            f"Rock #{parent_b.id} is {get_rock_gender_name(parent_b)}."
        )

    if block_siblings and game_are_siblings(parent_a, parent_b):
        errors.append("Sibling breeding is not allowed.")

    if block_parent_child and game_is_parent_child(parent_a, parent_b):
        errors.append("Parent-child breeding is not allowed.")

    if warn_related and game_are_related(game, parent_a, parent_b):
        if not game_are_siblings(parent_a, parent_b) and not game_is_parent_child(parent_a, parent_b):
            warnings.append("These rocks are related through shared ancestry.")

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "parent_a": parent_a,
        "parent_b": parent_b,
    }

def make_child_genome(parent_a, parent_b, mutation_rate=0.02):
    """
    Make a child's genome from two parents.

    Uses all genes present in either parent.
    """
    child_genes = {}

    all_gene_names = sorted(set(parent_a.genes.keys()) | set(parent_b.genes.keys()))

    for gene_name in all_gene_names:
        gene_a = parent_a.genes.get(gene_name, None)
        gene_b = parent_b.genes.get(gene_name, None)

        # If one parent somehow lacks a gene, roll from the other parent twice.
        if gene_a is None:
            gene_a = gene_b
        if gene_b is None:
            gene_b = gene_a

        child_genes[gene_name] = inherit_gene_from_parents(
            gene_name,
            gene_a,
            gene_b,
            mutation_rate=mutation_rate
        )

    return child_genes

def breed_child_for_game(
    game,
    parent_a_id,
    parent_b_id,
    mutation_rate=0.02,
    child_name=None,
    Not_importing = True,
    negative_id = None
):
    """
    Breed one child from a valid parent pair.

    Does not advance generation by itself.
    """
    result = validate_breeding_pair(game, parent_a_id, parent_b_id)

    if not result["valid"]:
        raise ValueError("Invalid breeding pair: " + "; ".join(result["errors"]))

    parent_a = result["parent_a"]
    parent_b = result["parent_b"]

    if Not_importing == True:
        child_id = game.next_id
        game.next_id += 1
    else:
        child_id = negative_id

    child_genes = make_child_genome(
        parent_a,
        parent_b,
        mutation_rate=mutation_rate
    )

    child = Rock(
        id=child_id,
        name=child_name if child_name is not None else random_rock_name(),
        genes=child_genes,
        parents=(parent_a.id, parent_b.id),
        generation=game.generation + 1
    )

    ensure_rock_game_attributes(child, imported=False, sold=False)
    evaluate_rock_value(child)

    if Not_importing == True:
        game.rocks[child_id] = child

    return child

def are_related(game, rock_a, rock_b):
    """
    Broad relationship check:
    siblings, parent-child, or shared ancestor.
    """
    if are_siblings(rock_a, rock_b):
        return True

    if is_parent_child(rock_a, rock_b):
        return True

    ancestors_a = get_ancestors(game, rock_a.id)
    ancestors_b = get_ancestors(game, rock_b.id)

    if rock_a.id in ancestors_b or rock_b.id in ancestors_a:
        return True

    return len(ancestors_a & ancestors_b) > 0

def anti_craisen_reroll_child_if_needed(
    game,
    child,
    parent_a_id,
    parent_b_id,
    mutation_rate,
    max_rerolls=ANTI_CRAISEN_REROLLS
):
    """
    Anti-Craisen Potion:
    if child is craisen, reroll its genome up to max_rerolls times.

    Keeps same child ID/name/parents/generation.
    """
    evaluate_rock_value(child)

    if getattr(child, "is_craisen", 0) != 1:
        return 0

    parent_a = get_rock(game, parent_a_id)
    parent_b = get_rock(game, parent_b_id)

    rerolls_used = 0

    for _ in range(max_rerolls):
        rerolls_used += 1

        child.genes = make_child_genome(
            parent_a,
            parent_b,
            mutation_rate=mutation_rate
        )

        # Reset status before re-evaluation.
        child.dead = False
        child.puffed = False
        child.death_reason = None
        child.rock_cost = 0
        child.is_craisen = 0

        evaluate_rock_value(child)

        if getattr(child, "is_craisen", 0) != 1:
            break

    return rerolls_used

def normalize_potion_key(potion_key):
    """
    Convert UI potion values into clean keys.
    """
    if potion_key in [None, "none", "None", ""]:
        return None

    return str(potion_key)

def get_potion_name(potion_key):
    potion_key = normalize_potion_key(potion_key)

    if potion_key is None:
        return "No potion"

    return POTION_SHOP.get(potion_key, {}).get("name", potion_key)

def get_owned_potion_options(game):
    """
    Options for the potion-application dropdown.
    Only owned potions appear.
    """
    options = [("No potion", None)]

    for potion_key, count in sorted(game.potions.items()):
        if count <= 0:
            continue

        name = get_potion_name(potion_key)
        options.append((f"{name} x{count}", potion_key))

    return options

def consume_potion(game, potion_key):
    """
    Consume one potion from inventory.
    """
    potion_key = normalize_potion_key(potion_key)

    if potion_key is None:
        return True

    if game.potions.get(potion_key, 0) <= 0:
        print(f"You do not own {get_potion_name(potion_key)}.")
        return False

    game.potions[potion_key] -= 1

    if game.potions[potion_key] <= 0:
        del game.potions[potion_key]

    return True

def refund_potion(game, potion_key):
    """
    Refund one potion back into inventory.
    """
    potion_key = normalize_potion_key(potion_key)

    if potion_key is None:
        return

    game.potions[potion_key] = game.potions.get(potion_key, 0) + 1

def make_breeding_queue_entry(parent_a_id, parent_b_id, potion_key=None):
    """
    New queue entry format.
    """
    return {
        "parents": (int(parent_a_id), int(parent_b_id)),
        "potion": normalize_potion_key(potion_key)
    }

def get_queue_entry_pair(entry):
    """
    Supports old tuple queue entries and new dict entries.
    """
    if isinstance(entry, dict):
        return entry["parents"]

    return entry

def get_queued_parent_ids(game):
    """
    Return all rock IDs currently used in the breeding queue.
    """
    queued = set()

    if game is None:
        return queued

    if not hasattr(game, "breeding_queue") or game.breeding_queue is None:
        return queued

    for entry in game.breeding_queue:
        try:
            a, b = get_queue_entry_pair(entry)
            queued.add(int(a))
            queued.add(int(b))
        except Exception:
            continue

    #print("pog")

    return queued

def is_rock_queued_for_breeding(game, rock_id):
    """
    True if this rock is already in the current generation's breeding queue.
    """
    if rock_id is None:
        return False
    
    #print("frog")

    return int(rock_id) in get_queued_parent_ids(game)

def get_queue_labels_by_rock(game):
    """
    Return mapping:
        rock_id -> ["❤1", "❤2", ...]

    Each queued pair gets a number based on queue position.
    """
    labels = {}

    if game is None:
        return labels

    if not hasattr(game, "breeding_queue") or game.breeding_queue is None:
        return labels

    for i, entry in enumerate(game.breeding_queue, start=1):
        try:
            a, b = get_queue_entry_pair(entry)
        except Exception:
            continue

        label = f"❤{i}"

        labels.setdefault(int(a), []).append(label)
        labels.setdefault(int(b), []).append(label)

    return labels

def get_queue_entry_potion(entry):
    """
    Supports old tuple queue entries and new dict entries.
    """
    if isinstance(entry, dict):
        return normalize_potion_key(entry.get("potion", None))

    return None

    """
    Return mapping:
        rock_id -> ["❤1", "❤2", ...]

    Each queued pair gets a number based on queue position.
    """
    labels = {}

    if game is None:
        return labels

    if not hasattr(game, "breeding_queue") or game.breeding_queue is None:
        return labels

    for i, entry in enumerate(game.breeding_queue, start=1):
        try:
            a, b = get_queue_entry_pair(entry)
        except Exception:
            continue

        label = f"❤{i}"

        labels.setdefault(int(a), []).append(label)
        labels.setdefault(int(b), []).append(label)

    return labels

def remove_selected_pairs_from_breeding_queue(game, pair_indices):
    """
    Remove selected queued breeding pairs by index.

    Important:
    remove from highest index to lowest index so earlier removals
    do not shift later indexes.

    Uses remove_pair_from_breeding_queue(), so attached potions are refunded.
    """
    if pair_indices is None:
        return 0

    cleaned_indices = sorted(
        set(int(i) for i in pair_indices),
        reverse=True
    )

    removed_count = 0

    for i in cleaned_indices:
        if 0 <= i < len(game.breeding_queue):
            if remove_pair_from_breeding_queue(game, i):
                removed_count += 1

    return removed_count

POTION_SHOP = {
    "anti_craisen": {
        "name": "Anti-Craisen Potion",
        "cost": 5,
        "description": "Later: reduce or reroll craisen offspring risk."
    },
    "mutation": {
        "name": "Mutation Potion",
        "cost": 2,
        "description": "Later: increase mutation chance for one breeding pair."
    },
    "fertility": {
        "name": "Fertility Potion",
        "cost": 3,
        "description": "Later: produce extra child from one pair."
    },
    "reroll": {
        "name": "Reroll Potion",
        "cost": 3,
        "description": "Later: reroll clutch size from one pair."
    },
}

def get_unsold_score_value(game):
    """
    Value of unsold, unbred, non-craisen rocks.
    """
    evaluate_all_rocks(game)

    return sum(
        rock.score_value
        for rock in game.rocks.values()
        if not getattr(rock, "sold", False)
    )

def get_final_score_estimate(game):
    """
    Current score estimate.

    Cash still counts because it is money already secured.
    Unsold score value counts only for rocks that are:
    - unsold
    - non-craisen
    - not used as parents
    """
    return game.money + get_unsold_score_value(game)

def show_money_summary(game):
    """
    Print money and score information.
    """
    evaluate_all_rocks(game)

    print("====================================")
    print("MONEY SUMMARY")
    print("====================================")
    print(f"Cash money: ${game.money}")
    print(f"Unsold eligible rock value: ${get_unsold_score_value(game)}")
    print(f"Current score estimate: ${get_final_score_estimate(game)}")
    print("====================================")

def get_rock_render_signature(rock):
    """
    Stable signature for a rock's visual appearance.

    If genes change, signature changes.
    If ID changes, signature changes, which matters because many drawings
    use rock.id for deterministic variation.
    """
    payload = {
        "id": rock.id,
        "genes": rock.genes,
    }

    raw = json.dumps(payload, sort_keys=True)

    return hashlib.md5(raw.encode("utf-8")).hexdigest()

def ensure_render_cache(game):
    """
    Runtime-only image cache.
    Not saved to JSON.
    """
    if not hasattr(game, "render_cache"):
        game.render_cache = {}

    return game.render_cache

def rock_to_image_uri_cached(game, rock):
    """
    Cached rock image renderer.

    This avoids regenerating the Matplotlib/base64 image every rerun.
    """
    cache = ensure_render_cache(game)

    sig = get_rock_render_signature(rock)

    if sig not in cache:
        cache[sig] = rock_to_image_uri(rock)

    return cache[sig]

def clear_render_cache(game):
    """
    Useful if draw_rock changes while debugging.
    """
    game.render_cache = {}

def format_selected_phenotype_hover(rock):
    v = get_visual_phenotype(rock)

    preferred_keys = [
        "gender",
        "shape",
        "size",
        "color",
        "eyes",
        "eye_color",
        "mouths",
        "noses",
        "arms",
        "wings",
        "horns",
        "halos",
        "ears",
        "hair",
        "hair_color",
        "facial_hair",
        "wrinkles",
        "fuzz",
        "freckles",
        "stones",
        "tails",
        "splitting",
    ]

    lines = []

    for key in preferred_keys:
        if key in v:
            lines.append(f"{key}: {clean_hover_value(v[key])}")

    return "<br>".join(lines)



def normalize_parent_pair_for_tree(parent_pair):
    """
    Convert a parent-pair-like value into a clean (p1, p2) tuple.

    Returns None if the value is empty, malformed, or not exactly two ids.

    Handles:
        None
        ()
        []
        (1, 2)
        ["1", "2"]
        {"a": 1, "b": 2}
        Rock objects with .id
    """
    if parent_pair is None:
        return None

    if isinstance(parent_pair, dict):
        raw_values = list(parent_pair.values())
    elif isinstance(parent_pair, (list, tuple, set)):
        raw_values = list(parent_pair)
    else:
        raw_values = [parent_pair]

    cleaned = []

    for value in raw_values:
        if value is None:
            continue

        if hasattr(value, "id"):
            value = value.id

        try:
            cleaned.append(int(value))
        except Exception:
            continue

    if len(cleaned) != 2:
        return None

    p1, p2 = cleaned

    if p1 == p2:
        return None

    return (p1, p2)



def draw_game_tree(
    game,
    selected_ids=None,
    x_gap=3.2,
    gen_gap=3.2,
    parent_pair_gap=1.7,
    rock_image_size=1.15,
    canvas_width=1800,
    canvas_height=1100,
    show_labels=True,
    show_sold=True,
    inactive_sold_opacity=0.55,
    show = False,
    highlight_breeding_queue=False
):
    """
    Draw the full game lineage tree.

    Features:
    - all rocks shown
    - sold rocks marked with green $
    - craisen rocks marked with red X
    - bred parents marked with gray circle if not sold/craisen
    - selected rocks highlighted
    """
    evaluate_all_rocks(game)

    selected_ids = set(selected_ids or [])

    rocks_dict = game.rocks

    if len(rocks_dict) == 0:
        print("No rocks to draw.")
        return None

    pos = compute_lineage_positions_christmas(
        rocks_dict,
        x_gap=x_gap,
        gen_gap=gen_gap,
        parent_pair_gap=parent_pair_gap,
        anti_overlap=True
    )

    fig = go.Figure()

    # Group children by parent pair.
    families = {}

    for child_id, rock in rocks_dict.items():
        if rock.parents is not None:
            key = tuple(sorted(rock.parents))
            families.setdefault(key, []).append(child_id)

    family_styles = get_family_styles(rocks_dict)

    # Draw family lines.
    for parent_pair, child_ids in sorted(families.items()):
        parent_pair = normalize_parent_pair_for_tree(parent_pair)

        if parent_pair is None:
            continue

        p1, p2 = parent_pair

        if p1 not in pos or p2 not in pos:
            continue

        x_line, y_line = build_family_segments(pos, parent_pair, child_ids)

        if len(x_line) == 0:
            continue

        style = family_styles.get(parent_pair, {"color": "#4E79A7", "dash": "solid"})

        fig.add_trace(
            go.Scatter(
                x=x_line,
                y=y_line,
                mode="lines",
                line=dict(
                    width=7,
                    color="rgba(255,255,255,0.95)",
                    dash=style["dash"]
                ),
                hoverinfo="skip",
                showlegend=False
            )
        )

        fig.add_trace(
            go.Scatter(
                x=x_line,
                y=y_line,
                mode="lines",
                line=dict(
                    width=3,
                    color=style["color"],
                    dash=style["dash"]
                ),
                hoverinfo="skip",
                showlegend=False
            )
        )

    # Add rock images.
    image_cache = {}

    for rid, rock in rocks_dict.items():
        if rid not in pos:
            continue

        x, y = pos[rid]

        if rid not in image_cache:
            image_cache[rid] = rock_to_image_uri_cached(game, rock)

        size_scale = get_rock_size_scale(rock)

        opacity = 1.0
        if getattr(rock, "sold", False):
            opacity = inactive_sold_opacity

        fig.add_layout_image(
            dict(
                source=image_cache[rid],
                xref="x",
                yref="y",
                x=x,
                y=y,
                sizex=rock_image_size * size_scale,
                sizey=rock_image_size * size_scale,
                xanchor="center",
                yanchor="middle",
                layer="above",
                opacity=opacity
            )
        )

    # Selected rings.
    for rid in selected_ids:
        if rid not in pos:
            continue

        x, y = pos[rid]

        fig.add_trace(
            go.Scatter(
                x=[x],
                y=[y],
                mode="markers",
                marker=dict(
                    size=rock_image_size * 85,
                    color="rgba(255,255,255,0)",
                    line=dict(
                        color="gold",
                        width=6
                    )
                ),
                hoverinfo="skip",
                showlegend=False
            )
        )

    # Hover points, labels, and status symbols.
    hover_x = []
    hover_y = []
    hover_text = []
    labels = []

    status_x = []
    status_y = []
    status_text = []
    status_colors = []

    for rid, rock in rocks_dict.items():
        if rid not in pos:
            continue

        x, y = pos[rid]
        v = get_visual_phenotype(rock)

        parent_text = "Founder/import"
        if rock.parents is not None:
            parent_text = f"Parents: #{rock.parents[0]} and #{rock.parents[1]}"

        flags = []
        if getattr(rock, "sold", False):
            flags.append("SOLD")
        if getattr(rock, "used_as_parent", False):
            flags.append("BRED PARENT")
        if getattr(rock, "is_craisen", 0) == 1:
            flags.append("CRAISEN")
        if getattr(rock, "imported", False):
            flags.append("IMPORTED")

        flag_text = ", ".join(flags) if flags else "OK"

        full_phenotype_text = format_selected_phenotype_hover(rock)

        text = (
            f"<b>{rock.name} #{rock.id}</b><br>"
            f"Generation: {rock.generation}<br>"
            f"{parent_text}<br>"
            f"Gender: {v.get('gender', 'n/a')} {get_gender_symbol(rock)}<br>"
            f"Base value: ${rock.base_value}<br>"
            f"Sell value: ${rock.sell_value}<br>"
            f"Score value: ${rock.score_value}<br>"
            f"Status: {flag_text}<br>"
            f"<br>"
            f"<b>Full phenotype</b><br>"
            f"{full_phenotype_text}"
        )

        hover_x.append(x)
        hover_y.append(y)
        hover_text.append(text)

        if show_labels:
            labels.append(f"{rock.name}<br>#{rock.id}")
        else:
            labels.append("")

        symbol = get_rock_status_symbol(rock)

        if symbol != "":
            status_x.append(x)
            status_y.append(y - 0.72 * rock_image_size)
            status_text.append(symbol)
            status_colors.append(get_rock_status_color(rock))

            # Queued breeding pair markers.
        if highlight_breeding_queue and len(game.breeding_queue) > 0:
            queue_labels_by_rock = get_queue_labels_by_rock(game)

            # Draw dashed red lines between currently queued future parents.
            for i, entry in enumerate(game.breeding_queue, start=1):
                a, b = get_queue_entry_pair(entry)

                if a not in pos or b not in pos:
                    continue

                xa, ya = pos[a]
                xb, yb = pos[b]

                # Slightly above rocks so it reads as a planned pair, not lineage.
                ya2 = ya + 0.78 * rock_image_size
                yb2 = yb + 0.78 * rock_image_size

                fig.add_trace(
                    go.Scatter(
                        x=[xa, xb],
                        y=[ya2, yb2],
                        mode="lines",
                        line=dict(
                            color="crimson",
                            width=3,
                            dash="dot"
                        ),
                        hoverinfo="skip",
                        showlegend=False
                    )
                )

                mid_x = 0.5 * (xa + xb)
                mid_y = 0.5 * (ya2 + yb2)

                fig.add_trace(
                    go.Scatter(
                        x=[mid_x],
                        y=[mid_y + 0.12 * rock_image_size],
                        mode="text",
                        text=[f"❤{i}"],
                        textfont=dict(
                            size=28,
                            color="crimson",
                            family="Arial Black"
                        ),
                        hoverinfo="skip",
                        showlegend=False
                    )
                )

            # Draw heart labels on each queued rock.
            for rid, labels in queue_labels_by_rock.items():
                if rid not in pos:
                    continue

                x, y = pos[rid]
                rock = rocks_dict[rid]
                size_scale = get_rock_size_scale(rock)

                fig.add_trace(
                    go.Scatter(
                        x=[x - 0.42 * rock_image_size * size_scale],
                        y=[y + 0.42 * rock_image_size * size_scale],
                        mode="text",
                        text=[" ".join(labels)],
                        textfont=dict(
                            size=24,
                            color="crimson",
                            family="Arial Black"
                        ),
                        hoverinfo="skip",
                        showlegend=False
                    )
                )

    # Invisible hover/labels.
    fig.add_trace(
        go.Scatter(
            x=hover_x,
            y=hover_y,
            mode="markers+text" if show_labels else "markers",
            marker=dict(
                size=rock_image_size * 48,
                color="rgba(0,0,0,0)"
            ),
            text=labels,
            textposition="bottom center",
            textfont=dict(size=10, color="black"),
            hovertext=hover_text,
            hoverinfo="text",
            showlegend=False
        )
    )

    # Status symbols.
    for sx, sy, st, sc in zip(status_x, status_y, status_text, status_colors):
        fig.add_trace(
            go.Scatter(
                x=[sx],
                y=[sy],
                mode="text",
                text=[st],
                textfont=dict(
                    size=30,
                    color=sc,
                    family="Arial Black"
                ),
                hoverinfo="skip",
                showlegend=False
            )
        )

    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]

        # Gender symbols near top-right of each rock.
    gender_x = []
    gender_y = []
    gender_text = []
    gender_colors = []

    for rid, rock in rocks_dict.items():
        if rid not in pos:
            continue

        x, y = pos[rid]

        size_scale = get_rock_size_scale(rock)

        gender_x.append(x + 0.42 * rock_image_size * size_scale)
        gender_y.append(y + 0.42 * rock_image_size * size_scale)
        gender_text.append(get_gender_symbol(rock))
        gender_colors.append(get_gender_color(rock))

    for gx, gy, gt, gc in zip(gender_x, gender_y, gender_text, gender_colors):
        fig.add_trace(
            go.Scatter(
                x=[gx],
                y=[gy],
                mode="text",
                text=[gt],
                textfont=dict(
                    size=26,
                    color=gc,
                    family="Arial Black"
                ),
                hoverinfo="skip",
                showlegend=False
            )
        )

    margin_x = 3.0
    margin_y = 3.0

    fig.update_layout(
        title=(
            f"Rock Game Tree — Generation {game.generation}/{game.max_generation} "
            f"| Cash ${game.money} | Score Estimate ${get_final_score_estimate(game)}"
        ),
        width=canvas_width,
        height=canvas_height,
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=dict(
            visible=False,
            range=[min(xs) - margin_x, max(xs) + margin_x]
        ),
        yaxis=dict(
            visible=False,
            range=[min(ys) - margin_y, max(ys) + margin_y],
            scaleanchor="x",
            scaleratio=1
        ),
        margin=dict(l=20, r=20, t=70, b=20),
        dragmode="pan"
    )

    if show:
        fig.show(config={
            "scrollZoom": True,
            "displayModeBar": True,
            "responsive": True
        })

    return fig

def pair_already_in_queue(game, parent_a_id, parent_b_id):
    """
    Check if pair is already in breeding queue.
    Order does not matter.
    """
    pair = tuple(sorted((int(parent_a_id), int(parent_b_id))))

    existing_pairs = [
        tuple(sorted(get_queue_entry_pair(entry)))
        for entry in game.breeding_queue
    ]

    return pair in existing_pairs

def add_pair_to_breeding_queue(game, parent_a_id, parent_b_id, potion_key=None):
    """
    Validate and add a pair to this generation's breeding queue.

    If potion_key is supplied, consume that potion immediately and attach it
    to this queued pair.
    """
    potion_key = normalize_potion_key(potion_key)

    if game.game_over:
        print("Game is over. No more breeding allowed.")
        return False

    if game.generation >= game.max_generation:
        print("Maximum generation reached. No more breeding allowed.")
        return False

    if len(game.breeding_queue) >= game.max_pairs_per_generation:
        print(
            f"Breeding queue is full: "
            f"{len(game.breeding_queue)} / {game.max_pairs_per_generation}"
        )
        return False

    if pair_already_in_queue(game, parent_a_id, parent_b_id):
        print("That pair is already in the breeding queue.")
        return False

    queued_ids = get_queued_parent_ids(game)

    if int(parent_a_id) in queued_ids:
        print(f"Rock #{parent_a_id} is already queued for breeding this generation.")
        return False

    if int(parent_b_id) in queued_ids:
        print(f"Rock #{parent_b_id} is already queued for breeding this generation.")
        return False

    result = validate_breeding_pair(game, parent_a_id, parent_b_id)

    if not result["valid"]:
        print("Cannot add invalid pair.")
        for error in result["errors"]:
            print(f"- {error}")
        return False

    if potion_key is not None:
        if not consume_potion(game, potion_key):
            print("Pair was not added because the potion could not be consumed.")
            return False

    entry = make_breeding_queue_entry(parent_a_id, parent_b_id, potion_key=potion_key)
    game.breeding_queue.append(entry)

    potion_text = ""
    if potion_key is not None:
        potion_text = f" using {get_potion_name(potion_key)}"

    game.events.append(
        f"Added breeding pair #{parent_a_id} x #{parent_b_id}{potion_text}. \n {format_relatedness_report(game, parent_a_id, parent_b_id)} "
    )

    print(
        f"Added pair #{parent_a_id} x #{parent_b_id}{potion_text} "
        f"to queue ({len(game.breeding_queue)} / {game.max_pairs_per_generation})."
    )

    return True

def remove_pair_from_breeding_queue(game, pair_index):
    """
    Remove a queued pair and refund its potion if it had one.
    """
    if pair_index < 0 or pair_index >= len(game.breeding_queue):
        print("Invalid pair index.")
        return False

    removed = game.breeding_queue.pop(pair_index)

    a, b = get_queue_entry_pair(removed)
    potion_key = get_queue_entry_potion(removed)

    refund_potion(game, potion_key)

    game.events.append(
        f"Removed breeding pair #{a} x #{b}."
    )

    if potion_key is not None:
        print(f"Removed pair #{a} x #{b}. Refunded {get_potion_name(potion_key)}.")
    else:
        print(f"Removed pair #{a} x #{b}.")

    return True

def clear_breeding_queue(game):
    """
    Clear queue and refund any attached potions.
    """
    for entry in game.breeding_queue:
        potion_key = get_queue_entry_potion(entry)
        refund_potion(game, potion_key)

    game.breeding_queue = []
    game.events.append("Cleared breeding queue.")
    print("Breeding queue cleared. Attached potions were refunded.")

def show_breeding_queue(game):
    """
    Print current breeding queue.
    """
    print("====================================")
    print("BREEDING QUEUE")
    print("====================================")
    print(f"Generation: {game.generation} / {game.max_generation}")
    print(f"Pairs: {len(game.breeding_queue)} / {game.max_pairs_per_generation}")
    print("------------------------------------")

    if len(game.breeding_queue) == 0:
        print("No pairs queued.")
    else:
        for i, entry in enumerate(game.breeding_queue):
            a, b = get_queue_entry_pair(entry)
            potion_key = get_queue_entry_potion(entry)

            rock_a = get_rock(game, a)
            rock_b = get_rock(game, b)

            name_a = rock_a.name if rock_a is not None else "missing"
            name_b = rock_b.name if rock_b is not None else "missing"

            potion_text = get_potion_name(potion_key)

            print(f"{i}: #{a} {name_a}  x  #{b} {name_b} | Potion: {potion_text}")

    print("====================================")

def validate_breeding_queue(game):
    """
    Validate every pair currently in the breeding queue.

    Supports both:
    - old tuple entries: (a, b)
    - new dict entries: {"parents": (a, b), "potion": potion_key}
    """
    queue_report = []
    all_valid = True

    if not hasattr(game, "breeding_queue"):
        game.breeding_queue = []

    for i, entry in enumerate(game.breeding_queue):
        a, b = get_queue_entry_pair(entry)
        potion_key = get_queue_entry_potion(entry)

        result = validate_breeding_pair(game, a, b)

        queue_report.append({
            "index": i,
            "entry": entry,
            "pair": (a, b),
            "potion": potion_key,
            "result": result
        })

        if not result["valid"]:
            all_valid = False

    return all_valid, queue_report

def print_queue_validation_report(game):
    all_valid, report = validate_breeding_queue(game)

    print("====================================")
    print("QUEUE VALIDATION")
    print("====================================")

    if len(report) == 0:
        print("Breeding queue is empty.")
        print("====================================")
        return all_valid, report

    for item in report:
        i = item["index"]
        a, b = item["pair"]
        potion_key = item["potion"]
        result = item["result"]

        status = "VALID" if result["valid"] else "INVALID"

        print(f"{i}: #{a} x #{b} | Potion: {get_potion_name(potion_key)} -> {status}")

        for error in result.get("errors", []):
            print(f"   ERROR: {error}")

        for warning in result.get("warnings", []):
            print(f"   WARNING: {warning}")

    print("====================================")

    return all_valid, report

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
import random
import numpy as np
import math

# ============================================================
# PLAYER PROFILE / CABAL CURSE GROUNDWORK
# ============================================================
CURSED_PLAYER_NAMES = {"tristan", "t"}

def normalize_player_name(name):
    """
    Normalize player names for save files and curse checks.
    """
    return str(name or "").strip().lower()

def sanitize_player_name_for_file(name):
    """
    Make a safe player name for save filenames.
    """
    name = str(name or "").strip()

    if name == "":
        return "unnamed_player"

    safe_chars = []

    for ch in name:
        if ch.isalnum():
            safe_chars.append(ch.lower())
        elif ch in [" ", "-", "_"]:
            safe_chars.append("_")

    safe = "".join(safe_chars)

    while "__" in safe:
        safe = safe.replace("__", "_")

    safe = safe.strip("_")

    if safe == "":
        return "unnamed_player"

    return safe

def get_game_save_filename(game):
    """
    Build player-specific save filename.
    """
    player_slug = sanitize_player_name_for_file(
        getattr(game, "player_name", "")
    )

    generation = int(getattr(game, "generation", 0))

    return f"rocks_{player_slug}_gen{generation}.json"

def ensure_player_profile_state(game):
    """
    Keep old saves compatible after adding player_name.
    """
    if not hasattr(game, "player_name") or game.player_name is None:
        game.player_name = ""

    if not hasattr(game, "cabal_curse_enabled") or game.cabal_curse_enabled is None:
        game.cabal_curse_enabled = False

    return game

def set_player_name(game, player_name):
    """
    Store player name on the game.
    """
    ensure_player_profile_state(game)
    game.player_name = str(player_name or "").strip()
    return game

def is_cabal_cursed(game):
    """
    Curse groundwork.

    Currently only activates if:
    - cabal_curse_enabled is manually True
    OR
    - normalized player name is in CURSED_PLAYER_NAMES

    Since CURSED_PLAYER_NAMES is empty by default, nobody is targeted yet.
    """
    ensure_player_profile_state(game)

    if bool(getattr(game, "cabal_curse_enabled", False)):
        return True

    name = normalize_player_name(getattr(game, "player_name", ""))

    cursed_names = {
        normalize_player_name(n)
        for n in CURSED_PLAYER_NAMES
    }

    return name in cursed_names

STARTER_GENDERS = {
    1: ("Male", "01"),
    2: ("Female", "00"),
    3: ("Male", "01"),
    4: ("Female", "00"),
}

DEFAULT_STARTING_MONEY = 10
DEFAULT_MAX_GENERATION = 7
DEFAULT_MAX_PAIRS_PER_GENERATION = 3

if is_cabal_cursed:
    CHILD_DEATH_CHANCE = 0.08 
else:
    CHILD_DEATH_CHANCE = 0.05 
if is_cabal_cursed:
    CLUTCH_MEAN = 1.1
else:
    CLUTCH_MEAN = 1.5
if is_cabal_cursed:
    CLUTCH_STD = 1.5
else:
    CLUTCH_STD = 2.0
MAX_CLUTCH_SIZE = None    
SPORE_CLONE_COUNT = 3
if is_cabal_cursed:
    SPORE_PUFF_CHANCE = 0.35
else:
    SPORE_PUFF_CHANCE = 0.25

POTION_MUTATION_RATE = 0.12
FERTILITY_EXTRA_CHILDREN = 1
ANTI_CRAISEN_REROLLS = 2

RANDOM_IMPORT_COST = 8
CUSTOM_IMPORT_MULTIPLIER = 2.0
CUSTOM_IMPORT_MIN_COST = 8
IMPORT_ROCK_COST = RANDOM_IMPORT_COST

REQUESTED_IMPORT_BASE_COST = 8
REQUESTED_IMPORT_MULTIPLIER = 2.0

def rebuild_used_as_parent_flags(game):
    """
    Rebuild used_as_parent flags from existing child parent links.

    Any rock that appears as a parent of another rock gets used_as_parent=True.
    """
    for rock in game.rocks.values():
        ensure_rock_game_attributes(rock)
        rock.used_as_parent = False

    for child in game.rocks.values():
        if child.parents is not None:
            for parent_id in child.parents:
                if parent_id in game.rocks:
                    game.rocks[parent_id].used_as_parent = True

    return game

def mark_pair_as_used_as_parents(game, parent_a_id, parent_b_id):
    """
    Mark two rocks as having been used for breeding.
    """
    for rid in [parent_a_id, parent_b_id]:
        rock = get_rock(game, rid)
        if rock is not None:
            ensure_rock_game_attributes(rock)
            rock.used_as_parent = True

    return game

def evaluate_all_rocks(game):
    """
    Refresh all values.
    """
    rebuild_used_as_parent_flags(game)

    for rock in game.rocks.values():
        evaluate_rock_value(rock)

    return game

def rock_has_mitosion(rock):
    """
    True if the rock expresses splitting = mitosion.
    """
    try:
        v = get_visual_phenotype(rock)
        return v.get("splitting", "n/a") == "mitosion"
    except Exception:
        return False

def duplicate_rock_for_mitosion(game, original_rock):
    """
    Create a duplicate clone of a mitosion rock.

    Same genes, same parents, same generation.
    New ID and slightly modified name.
    """
    clone_id = game.next_id
    game.next_id += 1

    clone = Rock(
        id=clone_id,
        name=f"{original_rock.name}_Mito",
        genes=dict(original_rock.genes),
        parents=original_rock.parents,
        generation=original_rock.generation
    )

    ensure_rock_game_attributes(clone, imported=False, sold=False)
    evaluate_rock_value(clone)

    game.rocks[clone_id] = clone

    return clone

def maybe_puff_spore_clone(clone, puff_chance=SPORE_PUFF_CHANCE):
    """
    A spore clone has a chance to puff out.

    Puffed clones:
    - remain visible
    - are worth $0
    - cannot breed
    """
    ensure_rock_game_attributes(clone)

    if random.random() < puff_chance:
        clone.puffed = True
        clone.dead = True
        clone.death_reason = "puffed out from spore"
        clone.sell_value = 0
        clone.score_value = 0
        return True

    return False

def create_spore_clone(game, original_rock, clone_index, puff_chance=SPORE_PUFF_CHANCE):
    """
    Create one spore clone from a spore-expressing rock.

    Same genes, same parents, same generation.
    """
    clone_id = game.next_id
    game.next_id += 1

    clone = Rock(
        id=clone_id,
        name=f"{original_rock.name}_Spore{clone_index}",
        genes=dict(original_rock.genes),
        parents=original_rock.parents,
        generation=original_rock.generation
    )

    ensure_rock_game_attributes(clone, imported=False, sold=False)

    puffed = maybe_puff_spore_clone(
        clone,
        puff_chance=puff_chance
    )

    evaluate_rock_value(clone)

    game.rocks[clone_id] = clone

    return clone, puffed

def create_spore_clones(
    game,
    original_rock,
    clone_count=SPORE_CLONE_COUNT,
    puff_chance=SPORE_PUFF_CHANCE
):
    """
    Spore behavior:
    produce several clones, each with a puff-out chance.
    """
    clones = []
    puffed_clones = []

    for i in range(1, clone_count + 1):
        clone, puffed = create_spore_clone(
            game,
            original_rock,
            clone_index=i,
            puff_chance=puff_chance
        )

        clones.append(clone)

        if puffed:
            puffed_clones.append(clone)

    return clones, puffed_clones

import math as m
# ============================================================
# BREEDING POD MARKET
# ============================================================

if is_cabal_cursed:
    low_cost_ceil = int(m.ceil(2 + random.random() * 1.3))
else:
    low_cost_ceil = int(m.ceil(2 + random.random() * 2))

if is_cabal_cursed:
    med_cost_ceil = int(m.ceil(5 + random.random() * 2.3))
else:
    med_cost_ceil = int(m.ceil(6 + random.random() * 3))

if is_cabal_cursed:
    high_cost_floor = int(m.ceil(4 + random.random() * 2))
else:
    high_cost_floor = int(m.ceil(6 + random.random() * 4))

MARKET_POD_TIERS = {
    "low": {
        "name": "Craigslist Gravel",
        "tagline": "Cheap, chaotic, and questionably damp.",
        "price": 3,
        "min_parent_value": 1,
        "max_parent_value": low_cost_ceil,
        "min_count": 0,
        "max_count": 2,
    },

    "medium": {
        "name": "Respectable Gravel",
        "tagline": "Decent family lines. Probably has a LinkedIn.",
        "price": 6,
        "min_parent_value": int(m.ceil(2 + random.random() * 2)),
        "max_parent_value": med_cost_ceil,
        "min_count": 0,
        "max_count": 2,
    },

    "high": {
        "name": "Boulder Elite",
        "tagline": "Pedigreed, polished, and financially insufferable.",
        "price": 10,
        "min_parent_value": high_cost_floor,
        "max_parent_value": 999,
        "min_count": 0,
        "max_count": 2,
    },
}


MARKET_GUEST_PARENT_SYMBOL = "NPC"

def get_current_breedable_generation(game):
    """
    The generation currently available to the player for breeding.

    Market children should enter this generation so they can be used
    immediately like imported rocks.
    """
    return int(getattr(game, "generation", 0))


def set_market_lineage_generations(game, parent_a, parent_b, child):
    """
    Place a kept market child in the current breedable generation.

    Guest parents are lineage-only rocks. They are shown as the child's
    parents, but they are not player-owned and cannot breed/sell.
    """
    current_gen = get_current_breedable_generation(game)

    # The kept child is an import into the current breeding pool.
    child.generation = current_gen

    # Guest parents should appear as lineage parents.
    # If we are past generation 0, put them one row above.
    # If current_gen is 0, keep them at 0 to avoid negative generation rows.
    parent_gen = max(0, current_gen - 1)

    parent_a.generation = parent_gen
    parent_b.generation = parent_gen

    return parent_a, parent_b, child



def force_rock_gender(rock, gender_name):
    """
    Force a rock to male or female.

    Game convention:
        female = 00
        male = 01
    """
    gender_name = str(gender_name).lower().strip()

    if gender_name in ["male", "m", "1"]:
        rock.genes["gender"] = "01"
        rock.gender = express_gender_from_gene("01")

    elif gender_name in ["female", "f", "0"]:
        rock.genes["gender"] = "00"
        rock.gender = express_gender_from_gene("00")

    else:
        raise ValueError(f"Unknown gender_name: {gender_name}")

    return rock


def reset_rock_as_market_founder(rock):
    """
    Market guest parents should be clean outside-founder rocks.

    They should not accidentally carry parent-child/sibling relationships.
    They should also not have empty parent tuples that confuse the tree.
    """
    # Use None for founder lineage, not empty tuples.
    rock.parents = None
    rock.parent_ids = None
    rock.parent_pair = None
    rock.parent_id_pair = None

    for attr in [
        "parent_a_id",
        "parent_b_id",
        "parent1_id",
        "parent2_id",
        "mother_id",
        "father_id",
        "mom_id",
        "dad_id",
        "parent_a",
        "parent_b",
        "mother",
        "father",
    ]:
        if hasattr(rock, attr):
            setattr(rock, attr, None)

    return rock




def ensure_market_state(game):
    """
    Ensure game has market fields.
    """
    if not hasattr(game, "market_pods") or game.market_pods is None:
        game.market_pods = []

    if not hasattr(game, "pending_market_pod"):
        game.pending_market_pod = None

    return game

def clear_market_state(game):
    """
    Clear current market pod offers and any unresolved pending pod.
    """
    game.market_pods = []
    game.pending_market_pod = None
    return game

def sync_next_rock_id(game):
    """
    Ensure game.next_id is always above every positive rock id currently in game.rocks.
    """
    if not hasattr(game, "rocks") or game.rocks is None:
        game.rocks = {}

    if not hasattr(game, "next_id") or game.next_id is None:
        game.next_id = 1

    used_ids = []

    for key, rock in game.rocks.items():
        try:
            used_ids.append(int(key))
        except Exception:
            pass

        try:
            used_ids.append(int(getattr(rock, "id", 0)))
        except Exception:
            pass

    positive_used_ids = [rid for rid in used_ids if rid > 0]

    if len(positive_used_ids) > 0:
        game.next_id = max(int(game.next_id), max(positive_used_ids) + 1)

    return game.next_id

def reserve_rock_id(game):
    """
    Reserve and return the next real positive rock id.
    """
    sync_next_rock_id(game)

    rid = int(game.next_id)
    game.next_id += 1

    return rid

def assign_new_rock_id(game, rock):
    """
    Assign a real unique positive id to a rock.
    """
    rock.id = reserve_rock_id(game)
    return rock.id

def add_rock_to_game_with_new_id(game, rock):
    """
    Assign a new unique id and add the rock to game.rocks.
    """
    assign_new_rock_id(game, rock)
    game.rocks[int(rock.id)] = rock
    return rock

def apply_requested_import_policy(rock, requested_values):
    """
    Apply requested-import logic to a random rock.

    Checked genes:
        force selected expression.

    Dependency-safe:
        eye_color ignored unless eyes requested.
        hair_color ignored unless hair/facial_hair/brows requested.
        hair_texture ignored unless hair/facial_hair requested.

    Unchecked catalog genes:
        suppress expression while preserving recessive carrier alleles where possible.
    """
    if requested_values is None:
        requested_values = {}

    # Backend safety: remove invalid dependent requests.
    filtered_requested_values = {}

    for gene_name, gene_value in requested_values.items():
        if requested_dependency_satisfied_from_values(gene_name, requested_values):
            filtered_requested_values[gene_name] = gene_value

    requested_values = filtered_requested_values

    actual_forced_values = {}

    for gene_name in REQUEST_TRAIT_CATALOG.keys():
        if gene_name not in rock.genes:
            continue

        if gene_name in requested_values:
            actual_value = make_requested_gene_value(
                gene_name,
                requested_values[gene_name]
            )

            rock.genes[gene_name] = actual_value
            actual_forced_values[gene_name] = actual_value

        else:
            rock.genes[gene_name] = suppress_gene_for_requested_import(
                gene_name,
                rock.genes[gene_name]
            )

    if "gender" in rock.genes:
        rock.gender = express_gender_from_gene(rock.genes["gender"])

    rock.requested_actual_forced_values = actual_forced_values

    return rock


def make_market_guest_parent_for_tier(game, tier_key, forced_gender=None, max_attempts=200):
    """
    Create a hidden guest parent for a market pod tier.

    Tier controls general value, not exact traits.
    forced_gender controls whether this guest parent is male/female.
    """
    tier = MARKET_POD_TIERS[tier_key]

    best_rock = None
    best_distance = float("inf")

    for _ in range(max_attempts):
        rock = make_random_rock(
            rock_id=-1,
            generation=getattr(game, "generation", 0)
        )

        reset_rock_as_market_founder(rock)

        if forced_gender is not None:
            force_rock_gender(rock, forced_gender)

        evaluate_rock_value(rock)

        value = getattr(rock, "sell_value", 0)

        if tier["min_parent_value"] <= value <= tier["max_parent_value"]:
            best_rock = rock
            break

        if value < tier["min_parent_value"]:
            distance = tier["min_parent_value"] - value
        elif value > tier["max_parent_value"]:
            distance = value - tier["max_parent_value"]
        else:
            distance = 0

        if distance < best_distance:
            best_distance = distance
            best_rock = rock

    # Safety pass after choosing best fallback.
    best_rock.id = -1
    reset_rock_as_market_founder(best_rock)

    if forced_gender is not None:
        force_rock_gender(best_rock, forced_gender)

    """
    best_rock.market_guest = True
    best_rock.owned = False
    best_rock.used_as_parent = False
    best_rock.sold = False
    best_rock.imported = True
    best_rock.market_tier = tier_key
    best_rock.market_guest_note = tier["name"]
    """

    evaluate_rock_value(best_rock)

    return best_rock

def generate_market_pods_for_generation(game, force=False):
    """
    Generate random pod offers for the current generation.

    Does not reveal parents in UI yet, but parents are pre-generated
    and stored in the offer.
    """
    ensure_market_state(game)

    if len(game.market_pods) > 0 and not force:
        return game.market_pods

    game.market_pods = []

    generation = getattr(game, "generation", 0) - 1

    for tier_key, tier in MARKET_POD_TIERS.items():
        count = random.randint(tier["min_count"], tier["max_count"])

        for i in range(count):
            # Make every pod a valid male/female pair.
            parent_a_gender = "male"
            parent_b_gender = "female"

            parent_a = make_market_guest_parent_for_tier(
                game,
                tier_key,
                forced_gender=parent_a_gender
            )

            parent_b = make_market_guest_parent_for_tier(
                game,
                tier_key,
                forced_gender=parent_b_gender
            )   
            """
            print(
                "Generated pod parents:",
                parent_a.genes.get("gender"),
                get_rock_gender_name(parent_a),
                "+",
                parent_b.genes.get("gender"),
                get_rock_gender_name(parent_b),
            )
            """
            offer = {
                "offer_id": f"pod_g{generation}_{tier_key}_{i}",
                "tier": tier_key,
                "name": tier["name"],
                "tagline": tier["tagline"],
                "price": tier["price"],
                "parent_a": parent_a,
                "parent_b": parent_b,
                "used": False,
            }

            game.market_pods.append(offer)

    #print(f"{game.market_pods[0]['parent_a']} + {game.market_pods[0]['parent_b']}")

    return game.market_pods

def breed_guest_parent_pair_for_market(game, parent_a, parent_b):
    """
    Breed two guest parents and return children WITHOUT adding all children
    permanently to the game.

    IMPORTANT:
    This function is the adapter between the market and your existing
    breeding engine.
    """

    created_children = []
    created_duplicates = []
    dead_children = []
    spore_events = []

    old_generation = game.generation - 1

    handle_mitosion, handle_sporing = True, True

    mark_pair_as_used_as_parents(game, parent_a.id, parent_b.id)

    clutch_size = roll_clutch_size()

    #print(parent_a.id)
    #print(parent_b.id)

    for _ in range(clutch_size):
            child = breed_child_for_game(
                game,
                parent_a.id,
                parent_b.id,
                Not_importing = False,
                negative_id= -1 * (
                        10000
                        + parent_a.id * 1000
                        + parent_b.id * 10
                        + _
                    )
                #mutation_rate=pair_mutation_rate
            )

            # Temporary candidate id. It is NOT a real family-tree id yet.
            

            created_children.append(child)

            died = maybe_kill_child(
                child,
                #death_chance=child_death_chance
            )

            evaluate_rock_value(child)

            if died:
                dead_children.append(child)
                game.events.append(
                    f"Child #{child.id} from #{parent_a} x #{parent_b} died after birth."
                )
                continue

            game.events.append(
                f"Generation {old_generation + 1}: bred #{parent_a} x #{parent_b} -> child #{child.id}."
            )

            try:
                v = get_visual_phenotype(child)
                splitting = v.get("splitting", "n/a")
            except Exception:
                splitting = "n/a"

            if handle_mitosion and splitting == "mitosion":
                clone = duplicate_rock_for_mitosion(game, child)
                created_duplicates.append(clone)

                game.events.append(
                    f"Rock #{child.id} expressed mitosion and duplicated into #{clone.id}."
                )

            elif splitting == "spore" and handle_sporing:
                clones, puffed_clones = create_spore_clones(
                    game,
                    child,
                    clone_count=SPORE_CLONE_COUNT,
                    puff_chance=SPORE_PUFF_CHANCE
                )

                spore_events.append({
                    "source": child,
                    "clones": clones,
                    "puffed": puffed_clones
                })

                created_duplicates.extend(clones)

                game.events.append(
                    f"Rock #{child.id} expressed spore and produced {len(clones)} spore clones; "
                    f"{len(puffed_clones)} puffed out."
                )

    
    return created_children + created_duplicates

def get_market_pod_offer(game, offer_id):
    ensure_market_state(game)

    for offer in game.market_pods:
        if offer["offer_id"] == offer_id:
            return offer

    return None

def buy_market_pod(game, offer_id):
    """
    Buy a market pod.

    Reveals guest parents and generated children.
    Does not finalize until player chooses one child.
    """
    ensure_market_state(game)

    if game.pending_market_pod is not None:
        print("You already have a pending market pod. Choose a child first.")
        return False

    offer = get_market_pod_offer(game, offer_id)

    if offer is None:
        print("Market pod not found.")
        return False

    if offer.get("used", False):
        print("That market pod has already been used.")
        return False

    price = int(offer["price"])

    if game.money < price:
        print(f"Not enough money. Need ${price}, but you have ${game.money}.")
        return False

    game.money -= price
    offer["used"] = True

    parent_a = offer["parent_a"]
    parent_b = offer["parent_b"]

    #print(parent_a.gender)
    #print(parent_b.gender)

    # Assign real ids now that they are entering the family tree.
    # Market parents become real only when the pod is bought.
    reset_rock_as_market_founder(parent_a)
    reset_rock_as_market_founder(parent_b)

    # Temporary generation. The final parent display generation is set
    # once the chosen child is known.
    parent_a.generation = max(0, get_current_breedable_generation(game) - 1)
    parent_b.generation = max(0, get_current_breedable_generation(game) - 1)

    add_rock_to_game_with_new_id(game, parent_a)
    add_rock_to_game_with_new_id(game, parent_b)


    children = breed_guest_parent_pair_for_market(
        game,
        parent_a,
        parent_b
    )

    # Children are not added to game.rocks yet.
    # The player gets to keep exactly one.
    for child in children:
        child.parent_ids = (parent_a.id, parent_b.id)
        child.parents = (parent_a.id, parent_b.id)
        child.market_child_candidate = True
        child.owned = False
        child.imported = True
        evaluate_rock_value(child)

    game.pending_market_pod = {
        "offer_id": offer_id,
        "tier": offer["tier"],
        "name": offer["name"],
        "price": price,
        "parent_a_id": parent_a.id,
        "parent_b_id": parent_b.id,
        "children": children,
    }

    print(f"You bought a {offer['name']} breeding pod for ${price}.")
    print("The guest parents have entered the tree.")
    print(f"They produced {len(children)} child candidate(s). Choose one to keep.")

    return True

def choose_market_pod_child(game, child_index):
    """
    Finalize pending market pod by keeping exactly one child.
    """
    ensure_market_state(game)

    pending = game.pending_market_pod

    if pending is None:
        print("No pending market pod child selection.")
        return False

    children = pending.get("children", [])

    if len(children) == 0:
        print("This market pod produced no children.")
        game.pending_market_pod = None
        return False

    child_index = int(child_index)

    if child_index < 0 or child_index >= len(children):
        print("Invalid child selection.")
        return False

    chosen_child = children[child_index]

    del(children)
    del(game.market_pods)

    assign_new_rock_id(game, chosen_child)

    chosen_child.owned = True
    chosen_child.market_child_candidate = False
    chosen_child.imported = True
    chosen_child.market_origin = pending["name"]

    # Make sure parent ids point to the guest parents.
    chosen_child.parent_ids = (
        int(pending["parent_a_id"]),
        int(pending["parent_b_id"]),
    )

    chosen_child.parents = chosen_child.parent_ids
    chosen_child.parent_pair = chosen_child.parent_ids
    chosen_child.parent_id_pair = chosen_child.parent_ids

    set_market_lineage_generations(
        game,
        game.rocks[int(pending["parent_a_id"])],
        game.rocks[int(pending["parent_b_id"])],
        chosen_child
    )

    game.rocks[chosen_child.id] = chosen_child

    evaluate_rock_value(chosen_child)

    game.pending_market_pod = None

    print(
        f"You kept #{chosen_child.id} {chosen_child.name} "
        f"from the {pending['name']} pod."
    )
    print("The other children returned to the breeder. Ruthless gravel capitalism.")

    return True





def create_new_game(
    starting_money=DEFAULT_STARTING_MONEY,
    max_generation=DEFAULT_MAX_GENERATION,
    max_pairs_per_generation=DEFAULT_MAX_PAIRS_PER_GENERATION,
    seed=None
):
    """
    Create a fresh 7-generation rock game.

    Starter rocks:
    #1 Male
    #2 Female
    #3 Male
    #4 Female
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    game = GameState(
        money=starting_money,
        max_generation=max_generation,
        max_pairs_per_generation=max_pairs_per_generation,
        generation=0,
        next_id=1,
        rocks={},
        breeding_queue=[],
        potions={},
        events=[],
        game_over=False,
        market_pods = [],
        pending_market_pod = None

    )

    for rid, (gender_name, gender_gene) in STARTER_GENDERS.items():
        rock = make_random_rock(rid)

        rock.genes["gender"] = gender_gene
        rock.gender = 1 if gender_gene == "01" else 0
        rock.generation = 0
        rock.parents = None

        rock.name = f"{gender_name}_{rock.name}"

        ensure_rock_game_attributes(
            rock,
            imported=False,
            sold=False
        )

        game.rocks[rid] = rock

    game.next_id = max(game.rocks.keys()) + 1

    evaluate_all_rocks(game)

    ensure_market_state(game)

    ensure_player_profile_state(game)

    game.events.append("New game started with 4 starter rocks.")

    return game

def activate_game(game):
    """
    Sync GameState into the global rocks/next_id variables
    used by the existing drawing and lineage tools.
    """
    global rocks, next_id

    rocks = game.rocks
    next_id = game.next_id

    return game

def sync_game_from_globals(game):
    """
    If older code updates global rocks/next_id, this pulls that back into game.
    """
    global rocks, next_id

    game.rocks = rocks
    game.next_id = next_id

    evaluate_all_rocks(game)

    return game

def get_active_rocks(game):
    return {
        rid: rock
        for rid, rock in game.rocks.items()
        if not getattr(rock, "sold", False)
    }

def get_sold_rocks(game):
    return {
        rid: rock
        for rid, rock in game.rocks.items()
        if getattr(rock, "sold", False)
    }

def get_craisen_rocks(game):
    return {
        rid: rock
        for rid, rock in game.rocks.items()
        if getattr(rock, "is_craisen", 0) == 1
    }

def show_game_status(game):
    """
    Print a compact run status.
    """
    evaluate_all_rocks(game)

    active = get_active_rocks(game)
    sold = get_sold_rocks(game)
    craisen = get_craisen_rocks(game)

    used_parents = {
        rid: rock
        for rid, rock in game.rocks.items()
        if getattr(rock, "used_as_parent", False)
    }

    total_unsold_value = get_unsold_score_value(game)

    print("====================================")
    print("ROCK GAME STATUS")
    print("====================================")
    print(f"Generation: {game.generation} / {game.max_generation}")
    print(f"Cash money: ${game.money}")
    print(f"Breeding queue: {len(game.breeding_queue)} / {game.max_pairs_per_generation}")
    print(f"Active rocks: {len(active)}")
    print(f"Sold rocks: {len(sold)}")
    print(f"Craisen rocks: {len(craisen)}")
    print(f"Bred parent rocks worth $0: {len(used_parents)}")
    print(f"Unsold eligible rock value: ${total_unsold_value}")
    print(f"Current score estimate: ${game.money + total_unsold_value}")
    print("====================================")

    if len(game.events) > 0:
        print("Recent events:")
        for event in game.events[-5:]:
            print(f"- {event}")

    print("====================================")

def show_game_rocks(game, cols=4, figsize_per_rock=4, include_sold=False):
    """
    Show current game rocks.
    """
    evaluate_all_rocks(game)

    if include_sold:
        rock_dict = game.rocks
        title = "All Game Rocks"
    else:
        rock_dict = get_active_rocks(game)
        title = "Active Game Rocks"

    show_rocks(
        rock_dict,
        cols=cols,
        figsize_per_rock=figsize_per_rock,
        show_traits=True,
        normalize_size=False,
        title=title,
        sort_by_generation=True
    )

def execute_breeding_generation(
    game,
    mutation_rate=0.02,
    child_death_chance=CHILD_DEATH_CHANCE,
    use_clutch_formula=True,
    fixed_children_per_pair=1,
    handle_mitosion=True,
    handle_sporing=True,
    abort_on_invalid=True,
    show_results=True
):
    """
    Execute the current breeding queue.

    Potion behavior:
    - anti_craisen: rerolls craisen child genomes up to ANTI_CRAISEN_REROLLS
    - mutation: increases mutation rate for that pair
    - fertility: adds FERTILITY_EXTRA_CHILDREN to that pair's clutch
    """
    if game.game_over:
        print("Game is already over.")
        return []

    if game.generation >= game.max_generation:
        game.game_over = True
        print("Maximum generation reached. Game over.")
        return []

    if len(game.breeding_queue) == 0:
        print("No breeding pairs queued.")
        return []

    all_valid, report = validate_breeding_queue(game)

    if not all_valid:
        print_queue_validation_report(game)

        if abort_on_invalid:
            print("Generation aborted because the queue contains invalid pairs.")
            return []

    created_children = []
    created_duplicates = []
    dead_children = []
    spore_events = []
    potion_events = []

    old_generation = game.generation

    for item in report:
        a, b = item["pair"]
        potion_key = item.get("potion", None)
        result = item["result"]

        if not result["valid"]:
            continue

        mark_pair_as_used_as_parents(game, a, b)

        pair_mutation_rate = get_pair_mutation_rate(mutation_rate, potion_key)

        if use_clutch_formula:
            base_clutch_size = roll_clutch_size()
        else:
            base_clutch_size = fixed_children_per_pair

        clutch_size = base_clutch_size
        clutch_size = apply_reroll_to_clutch(clutch_size, potion_key)
        clutch_size = apply_fertility_to_clutch(clutch_size, potion_key)

        if potion_key in ["reroll", "fertility"]:
            game.events.append(
                f"Generation {old_generation + 1}: pair #{a} x #{b} rolled clutch {base_clutch_size}, "
                f"final clutch {clutch_size} with {get_potion_name(potion_key)}."
            )
        else:
            game.events.append(
                f"Generation {old_generation + 1}: pair #{a} x #{b} rolled clutch size {clutch_size}."
            )

        for _ in range(clutch_size):
            child = breed_child_for_game(
                game,
                a,
                b,
                mutation_rate=pair_mutation_rate
            )

            if potion_key == "anti_craisen":
                rerolls = anti_craisen_reroll_child_if_needed(
                    game,
                    child,
                    a,
                    b,
                    mutation_rate=pair_mutation_rate,
                    max_rerolls=ANTI_CRAISEN_REROLLS
                )

                if rerolls > 0:
                    potion_events.append(
                        f"Anti-Craisen rerolled child #{child.id} {rerolls} time(s)."
                    )

            created_children.append(child)

            died = maybe_kill_child(
                child,
                death_chance=child_death_chance
            )

            evaluate_rock_value(child)

            if died:
                dead_children.append(child)
                game.events.append(
                    f"Child #{child.id} from #{a} x #{b} died after birth."
                )
                continue

            game.events.append(
                f"Generation {old_generation + 1}: bred #{a} x #{b} -> child #{child.id}."
            )

            try:
                v = get_visual_phenotype(child)
                splitting = v.get("splitting", "n/a")
            except Exception:
                splitting = "n/a"

            if handle_mitosion and splitting == "mitosion":
                clone = duplicate_rock_for_mitosion(game, child)
                created_duplicates.append(clone)

                game.events.append(
                    f"Rock #{child.id} expressed mitosion and duplicated into #{clone.id}."
                )

            elif splitting == "spore" and handle_sporing:
                clones, puffed_clones = create_spore_clones(
                    game,
                    child,
                    clone_count=SPORE_CLONE_COUNT,
                    puff_chance=SPORE_PUFF_CHANCE
                )

                spore_events.append({
                    "source": child,
                    "clones": clones,
                    "puffed": puffed_clones
                })

                created_duplicates.extend(clones)

                game.events.append(
                    f"Rock #{child.id} expressed spore and produced {len(clones)} spore clones; "
                    f"{len(puffed_clones)} puffed out."
                )

    game.generation += 1
    game.breeding_queue = []

    clear_market_state(game)
    generate_market_pods_for_generation(game, force=True)

    evaluate_all_rocks(game)

    if game.generation >= game.max_generation:
        game.game_over = True
        game.events.append("Final generation reached. Game over.")

    activate_game(game)

    if show_results:
        total_spore_clones = sum(len(event["clones"]) for event in spore_events)
        total_puffed_spores = sum(len(event["puffed"]) for event in spore_events)

        print("====================================")
        print("GENERATION EXECUTED")
        print("====================================")
        print(f"Advanced from Gen {old_generation} to Gen {game.generation}.")
        print(f"Children born: {len(created_children)}")
        print(f"Children died: {len(dead_children)}")
        print(f"Mitosion duplicates: {len([r for r in created_duplicates if not getattr(r, 'puffed', False)])}")
        print(f"Spore events: {len(spore_events)}")
        print(f"Spore clones: {total_spore_clones}")
        print(f"Puffed spore clones: {total_puffed_spores}")

        if len(potion_events) > 0:
            print("\nPotion events:")
            for event in potion_events:
                print(f"- {event}")

        if len(created_children) > 0:
            print("\nChildren:")
            for child in created_children:
                if getattr(child, "dead", False):
                    status = "DEAD"
                elif child.is_craisen:
                    status = "CRAISEN"
                else:
                    status = "OK"

                print(
                    f"- #{child.id} {child.name} | "
                    f"parents #{child.parents[0]} x #{child.parents[1]} | "
                    f"value ${child.sell_value} | {status}"
                )

        if len(spore_events) > 0:
            print("\nSpore events:")
            for event in spore_events:
                source = event["source"]
                clones = event["clones"]
                puffed = event["puffed"]

                print(
                    f"- #{source.id} {source.name} expressed spore: "
                    f"{len(clones)} clones, {len(puffed)} puffed."
                )

        print("====================================")

    return created_children + created_duplicates

def run_generation_from_ui(game):
    """
    UI-safe generation executor.

    Uses the newer clutch/death/spore generation system.
    """
    return execute_breeding_generation(
        game,
        mutation_rate=0.02,
        child_death_chance=CHILD_DEATH_CHANCE,
        use_clutch_formula=True,
        fixed_children_per_pair=1,
        handle_mitosion=True,
        handle_sporing=True,
        abort_on_invalid=True,
        show_results=True
    )

def is_rock_sold_flag(rock):
    return bool(getattr(rock, "sold", False))

def get_rock_status_symbol(rock):
    """
    Symbol shown near rocks in game views.
    """
    if getattr(rock, "puffed", False):
        return "☁"

    if getattr(rock, "dead", False):
        return "†"

    if is_rock_sold_flag(rock):
        return "$"

    if getattr(rock, "is_craisen", 0) == 1:
        return "X"

    if getattr(rock, "used_as_parent", False):
        return "○"

    if getattr(rock, "market_guest", False):
        return(("NPC", "darkviolet"))

    return ""

def get_rock_status_color(rock):
    """
    Color for status symbols.
    """
    if getattr(rock, "puffed", False):
        return "dimgray"

    if getattr(rock, "dead", False):
        return "black"

    if is_rock_sold_flag(rock):
        return "green"

    if getattr(rock, "is_craisen", 0) == 1:
        return "crimson"

    if getattr(rock, "used_as_parent", False):
        return "gray"

    return "black"

def sell_rock(game, rock_id, allow_zero_sale=False):
    """
    Sell a rock for cash.

    Bred parents, craisen rocks, and already-sold rocks have zero value.
    """
    rock = get_rock(game, rock_id)

    if rock is None:
        print(f"Rock #{rock_id} does not exist.")
        return False

    ensure_rock_game_attributes(rock)
    evaluate_rock_value(rock)

    if getattr(rock, "sold", False):
        print(f"Rock #{rock.id} has already been sold.")
        return False

    value = rock.sell_value

    if value <= 0 and not allow_zero_sale:
        print(f"Rock #{rock.id} has sell value $0 and was not sold.")
        if getattr(rock, "used_as_parent", False):
            print("Reason: this rock has already been used as a parent.")
        if getattr(rock, "is_craisen", 0) == 1:
            print("Reason: this rock is craisen.")
        return False

    rock.sold = True
    game.money += value

    game.events.append(
        f"Sold rock #{rock.id} {rock.name} for ${value}."
    )

    evaluate_all_rocks(game)

    print(f"Sold rock #{rock.id} {rock.name} for ${value}.")
    print(f"Cash money is now ${game.money}.")

    return True

def sell_many_rocks(game, rock_ids, allow_zero_sale=False):
    """
    Sell multiple rocks.
    """
    sold_count = 0

    for rid in rock_ids:
        if sell_rock(game, rid, allow_zero_sale=allow_zero_sale):
            sold_count += 1

    print(f"Sold {sold_count} rocks.")
    return sold_count

def sell_all_score_eligible_rocks(game):
    """
    Sell all rocks that currently have sell_value > 0.
    Useful at the end of the game.
    """
    evaluate_all_rocks(game)

    sellable_ids = [
        rid
        for rid, rock in game.rocks.items()
        if not getattr(rock, "sold", False)
        and rock.sell_value > 0
    ]

    return sell_many_rocks(game, sellable_ids)

def is_parent_dropdown_eligible(rock):
    """
    Rock can appear in parent selector only if it can still breed.
    """
    ensure_rock_game_attributes(rock)

    if getattr(rock, "sold", False):
        return False

    if getattr(rock, "dead", False):
        return False

    if getattr(rock, "puffed", False):
        return False

    if getattr(rock, "is_craisen", 0) == 1:
        return False

    if getattr(rock, "used_as_parent", False):
        return False

    return True

def is_sell_dropdown_eligible(rock):

    """
    Rock can appear in sell selector only if it can actually be sold for money.
    """
    ensure_rock_game_attributes(rock)

    if getattr(rock, "sold", False):
        return False

    if getattr(rock, "dead", False):
        return False

    if getattr(rock, "puffed", False):
        return False

    if getattr(rock, "is_craisen", 0) == 1:
        return False

    if getattr(rock, "used_as_parent", False):
        return False

    if getattr(rock, "sell_value", 0) <= 0:
        return False

    return True

def is_parent_dropdown_eligible(rock):
    """
    Rock can appear in parent selector only if it can still breed.
    """
    ensure_rock_game_attributes(rock)

    if getattr(rock, "sold", False):
        return False

    if getattr(rock, "dead", False):
        return False

    if getattr(rock, "puffed", False):
        return False

    if getattr(rock, "is_craisen", 0) == 1:
        return False

    if getattr(rock, "used_as_parent", False):
        return False

    return True

def is_sell_dropdown_eligible(rock):
    """
    Rock can appear in sell selector only if it can actually be sold for money.
    """
    ensure_rock_game_attributes(rock)

    if getattr(rock, "sold", False):
        return False

    if getattr(rock, "dead", False):
        return False

    if getattr(rock, "puffed", False):
        return False

    if getattr(rock, "is_craisen", 0) == 1:
        return False

    if getattr(rock, "used_as_parent", False):
        return False

    if getattr(rock, "sell_value", 0) <= 0:
        return False

    return True

def get_breeding_dropdown_options(game):
    """
    Parent selector options.

    Only shows rocks that are currently eligible to breed:
    - unsold
    - alive
    - not puffed
    - not craisen
    - not already used as parent
    - not already queued this generation
    """
    evaluate_all_rocks(game)

    queued_ids = get_queued_parent_ids(game)
    options = []

    for rid, rock in sorted(game.rocks.items()):
        if not is_parent_dropdown_eligible(rock):
            continue

        if rid in queued_ids:
            continue

        gender = get_rock_gender_name(rock)

        flags = []

        if getattr(rock, "imported", False):
            flags.append("IMPORTED")

        flag_text = f" [{', '.join(flags)}]" if flags else ""

        label = (
            f"#{rid} {rock.name} | {gender} | "
            f"Gen {rock.generation} | Sell ${rock.sell_value}{flag_text}"
        )

        options.append((label, rid))

    if len(options) == 0:
        options = [("No eligible breeding rocks", None)]

    return options

def get_sell_dropdown_options(game):
    """
    Sell selector options.

    Only shows rocks that can actually be sold for money.
    """
    evaluate_all_rocks(game)

    options = []

    for rid, rock in sorted(game.rocks.items()):
        if not is_sell_dropdown_eligible(rock):
            continue

        flags = []

        if getattr(rock, "imported", False):
            flags.append("IMPORTED")

        flag_text = f" [{', '.join(flags)}]" if flags else ""

        label = (
            f"#{rid} {rock.name} | "
            f"Gen {rock.generation} | "
            f"Sell ${rock.sell_value}{flag_text}"
        )

        options.append((label, rid))

    if len(options) == 0:
        options = [("No sellable rocks", None)]

    return options

REQUEST_TRAIT_CATALOG = {
    "gender": {
        "label": "Gender",
        "options": {
            "Female": "00",
            "Male": "01",
        },
    },

    "shape": {
        "label": "Shape",
        "options": {
            "Circle": "00",
            "Oval": "11",
            "Square": "22",
            "Triangle": "33",
            "Oblong": "44",
        },
    },

    "size": {
        "label": "Size",
        "options": {
            "Medium": "00",
            "Large": "11",
            "Small": "22",
            "Giant": "33",
            "Missized": "44",
        },
    },

    "color": {
        "label": "Body Color",
        "options": {
            "White": "00",
            "Black": "11",
            "Silver": "01",
            "Brown": "22",
            "Red": "33",
            "Yellow": "44",
            "Blue": "55",
            "Orange": "34",
            "Purple": "35",
            "Green": "45",
            "Patchwork": "66",
        },
    },

    "eyes": {
        "label": "Eyes",
        "options": {
            "No Eyes": "00",
            "One Eye": "01",
            "Two Eyes": "11",
        },
    },

    "eye_color": {
        "label": "Eye Color",
        "options": {
            "White": "00",
            "Black": "11",
            "Red": "22",
            "Green": "33",
            "Blue": "44",
            "Yellow": "55",
            "Evil": "66",
            "Purple": "77",
            "Callus": "88",
        },
    },

    "brows": {
        "label": "Brows",
        "options": {
            "No Brows": "00",
            "Brows": "11",
            "Eyelash": "22",
            "Unibrow": "33",
        },
    },

    "mouths": {
        "label": "Mouth",
        "options": {
            "No Mouth": "00",
            "Mouth": "11",
            "Smile": "22",
            "Chip": "33",
            "Smeagol": "44",
        },
    },

    "noses": {
        "label": "Nose",
        "options": {
            "No Nose": "00",
            "Nub": "11",
            "Honk": "22",
            "Holes": "33",
            "Concave": "44",
        },
    },

    "arms": {
        "label": "Arms",
        "options": {
            "No Arms": "00",
            "Normal Pair": "01",
            "Double Normal": "11",
            "Muscle Pair": "02",
            "Mixed Normal/Muscle": "12",
            "Double Muscle": "22",
        },
    },

    "crowns": {
        "label": "Crown",
        "options": {
            "No Crown": "00",
            "Small Crown": "11",
            "Medium Crown": "22",
            "Large Crown": "33",
            "Indent": "44",
        },
    },

    "wings": {
        "label": "Wings",
        "options": {
            "No Wings": "00",
            "Wings": "11",
        },
    },

    "halos": {
        "label": "Halo",
        "options": {
            "No Halo": "00",
            "Halo": "11",
        },
    },

    "horns": {
        "label": "Horns",
        "options": {
            "No Horns": "00",
            "Horns": "11",
        },
    },

    "wrinkles": {
        "label": "Wrinkles",
        "options": {
            "No Wrinkles": "00",
            "Wrinkles": "11",
        },
    },

    "fuzz": {
        "label": "Fuzz",
        "options": {
            "No Fuzz": "00",
            "Fuzz": "01",
            "Spiky": "11",
        },
    },

    "freckles": {
        "label": "Freckles",
        "options": {
            "No Freckles": "00",
            "Freckles": "11",
        },
    },

    "stones": {
        "label": "Ion Stone",
        "options": {
            "No Stone": "00",
            "Ion Stone": "11",
        },
    },

    "tails": {
        "label": "Tail",
        "options": {
            "No Tail": "00",
            "Tail": "11",
        },
    },

    "ears": {
        "label": "Ears",
        "options": {
            "No Ears": "00",
            "Antannae": "11",
            "Human": "22",
            "Ogre": "33",
            "Goblin": "44",
        },
    },

    "facial_hair": {
        "label": "Facial Hair",
        "options": {
            "None": "00",
            "Goatee": "11",
            "Beard": "22",
            "Pedo": "33",
            "Curly": "44",
            "Chapman": "55",
            "Sol": "66",
        },
    },

    "hair": {
        "label": "Hair",
        "options": {
            "No Hair": "00",
            "Hair": "01",
            "Double Hair": "11",
        },
    },

    "hair_color": {
        "label": "Hair Color",
        "options": {
            "White": "00",
            "Black": "11",
            "Silver": "01",
            "Brown": "22",
            "Blonde": "33",
            "Red": "44",
            "Pink": "55",
            "Blue": "66",
        },
    },

    "hair_texture": {
        "label": "Hair Texture",
        "options": {
            "Straight": "00",
            "Curly": "11",
        },
    },

    "splitting": {
        "label": "Splitting",
        "options": {
            "None": "00",
            "Mitosion": "11",
            "Spore": "22",
        },
    },
}

REQUEST_FORCE_00_IF_UNCHECKED = {
    "eyes",
    "hair",
    "arms",
    "fuzz",
}

COLOR_IF_UNCHECKED = {
    "hair_color",
    "color",
}

REQUEST_TRAIT_DEPENDENCIES = {
    "eye_color": {
        "mode": "any",
        "parents": [
            {
                "gene": "eyes",
                "label": "Eyes",
                "blocked_values": {"00"},
            }
        ],
        "message": "Requires Eyes",
    },

    "hair_color": {
        "mode": "any",
        "parents": [
            {
                "gene": "hair",
                "label": "Hair",
                "blocked_values": {"00"},
            },
            {
                "gene": "facial_hair",
                "label": "Facial Hair",
                "blocked_values": {"00"},
            },
            {
                "gene": "brows",
                "label": "Brows",
                "blocked_values": {"00"},
            },
        ],
        "message": "Requires Hair, Facial Hair, or Brows",
    },

    "hair_texture": {
        "mode": "any",
        "parents": [
            {
                "gene": "hair",
                "label": "Hair",
                "blocked_values": {"00"},
            },
            {
                "gene": "facial_hair",
                "label": "Facial Hair",
                "blocked_values": {"00"},
            },
            {
                "gene": "brows",
                "label": "Brows",
                "blocked_values": {"00"},
            },
        ],
        "message": "Requires Hair or Facial Hair",
    },
}

def requested_dependency_satisfied_from_values(gene_name, requested_values):
    """
    Backend dependency check using requested_values dictionary.

    Used by requested import logic and can also support UI logic.

    Examples:
    - eye_color requires eyes
    - hair_color requires hair OR facial_hair OR brows
    - hair_texture requires hair OR facial_hair
    """
    dependency = REQUEST_TRAIT_DEPENDENCIES.get(gene_name, None)

    if dependency is None:
        return True

    mode = dependency.get("mode", "any")
    parent_rules = dependency.get("parents", [])

    checks = []

    for rule in parent_rules:
        parent_gene = rule["gene"]
        blocked_values = set(str(v) for v in rule.get("blocked_values", set()))

        if parent_gene not in requested_values:
            checks.append(False)
            continue

        parent_value = str(requested_values[parent_gene])

        if parent_value in blocked_values:
            checks.append(False)
            continue

        checks.append(True)

    if mode == "all":
        return all(checks)

    return any(checks)

def requested_trait_dependency_satisfied(gene_name, request_widgets):
    """
    Return True if this gene's dependency is satisfied.

    Example:
        eye_color requires eyes checked and eyes != 00
        hair_color requires hair checked and hair != 00
    """
    dependency = REQUEST_TRAIT_DEPENDENCIES.get(gene_name, None)

    if dependency is None:
        return True

    parent_gene = dependency["parent"]

    if parent_gene not in request_widgets:
        return False

    parent_checkbox = request_widgets[parent_gene]["checkbox"]
    parent_dropdown = request_widgets[parent_gene]["dropdown"]

    if not parent_checkbox.value:
        return False

    parent_value = str(parent_dropdown.value)

    if parent_value in dependency.get("blocked_parent_values", set()):
        return False

    return True

def update_requested_trait_dependency_states(request_widgets):
    """
    Enable/disable dependent requested-import controls.

    If dependency is not satisfied:
    - uncheck the dependent trait
    - disable checkbox and dropdown
    """
    for gene_name, dependency in REQUEST_TRAIT_DEPENDENCIES.items():
        if gene_name not in request_widgets:
            continue

        controls = request_widgets[gene_name]

        checkbox = controls["checkbox"]
        dropdown = controls["dropdown"]
        note = controls.get("note", None)

        satisfied = requested_trait_dependency_satisfied(gene_name, request_widgets)

        checkbox.disabled = not satisfied
        dropdown.disabled = not satisfied

        if not satisfied:
            checkbox.value = False

        if note is not None:
            if satisfied:
                note.value = ""
            else:
                note.value = f"<span style='color:gray;'>({dependency['message']})</span>"

def suppress_gene_for_requested_import(gene_name, current_gene_value):
    """
    Suppress visible expression for an unchecked requested-import trait.

    Co-dominant/additive traits:
        force 00

    Most normal traits:
        force first allele to 0, keep second allele random
        example: 34 -> 04

    Death genes:
        leave untouched. We do not want the requested-import system
        accidentally modifying death genetics.
    """
    if gene_name in DEATH_GENES:
        return current_gene_value

    current_gene_value = str(current_gene_value)

    if gene_name in REQUEST_FORCE_00_IF_UNCHECKED:
        return "00"

    if gene_name in COLOR_IF_UNCHECKED:
        if current_gene_value.count('1') > 0:
          return current_gene_value.replace('1', '0')

    if len(current_gene_value) <= 1:
        return "0" + current_gene_value

    # Normal two-allele trait: force first allele to 0, keep second.
    return "0" + current_gene_value[1]

def apply_import_gene_overrides(rock, gene_overrides=None, force_gender=None):
    """
    Apply specific requested genes to an imported rock.

    gene_overrides example:
        {
            "shape": "33",
            "wings": "11",
            "hair": "11",
            "hair_color": "44"
        }

    force_gender:
        None, "male", "female", 1, or 0
    """
    if gene_overrides is None:
        gene_overrides = {}

    for gene_name, gene_value in gene_overrides.items():
        rock.genes[gene_name] = str(gene_value)

    if force_gender in ["male", "Male", 1]:
        rock.genes["gender"] = "01"
        rock.gender = 1

    elif force_gender in ["female", "Female", 0]:
        rock.genes["gender"] = "00"
        rock.gender = 0

    return rock

def calculate_custom_import_cost(rock):
    """
    Custom import cost:
    max(minimum cost, 2 × base value)
    """
    evaluate_rock_value(rock)

    return max(
        CUSTOM_IMPORT_MIN_COST,
        math.ceil(CUSTOM_IMPORT_MULTIPLIER * rock.base_value)
    )

def preview_custom_import_rock(
    game,
    gene_overrides=None,
    force_gender=None,
    reroll_if_craisen=True,
    max_attempts=100
):
    """
    Generate a preview quote for a custom imported rock.

    The rock is NOT added to the game yet.
    The quote is stored as game.pending_custom_import.

    Returns:
        quote dict with rock, cost, attempts, gene_overrides
    """
    if gene_overrides is None:
        gene_overrides = {}

    last_rock = None

    for attempt in range(1, max_attempts + 1):
        rock = make_random_rock(game.next_id)

        rock.parents = None
        rock.generation = game.generation
        rock.name = f"CustomImport_{rock.name}"

        apply_import_gene_overrides(
            rock,
            gene_overrides=gene_overrides,
            force_gender=force_gender
        )

        ensure_rock_game_attributes(
            rock,
            imported=True,
            sold=False
        )

        evaluate_rock_value(rock)

        last_rock = rock

        if not reroll_if_craisen:
            break

        if getattr(rock, "is_craisen", 0) != 1:
            break

    if last_rock is None:
        raise RuntimeError("Could not generate custom import preview.")

    if reroll_if_craisen and getattr(last_rock, "is_craisen", 0) == 1:
        print(
            f"Warning: custom import was still craisen after {max_attempts} attempts. "
            "This may mean your overrides force a bad death-gene combination."
        )

    cost = calculate_custom_import_cost(last_rock)

    quote = {
        "rock": last_rock,
        "cost": cost,
        "attempts": attempt,
        "gene_overrides": dict(gene_overrides),
        "force_gender": force_gender,
        "reroll_if_craisen": reroll_if_craisen,
    }

    game.pending_custom_import = quote

    print("====================================")
    print("CUSTOM IMPORT QUOTE")
    print("====================================")
    print(f"Rock: #{last_rock.id} {last_rock.name}")
    print(f"Generation: {last_rock.generation}")
    print(f"Base value: ${last_rock.base_value}")
    print(f"Custom import cost: ${cost}")
    print(f"Craisen: {bool(last_rock.is_craisen)}")
    print(f"Attempts: {attempt}")
    print("Overrides:")
    for k, v in gene_overrides.items():
        print(f"- {k}: {v}")
    print("====================================")

    return quote

def buy_custom_import_quote(game, quote=None):
    """
    Buy the currently previewed custom import.
    """
    if quote is None:
        quote = getattr(game, "pending_custom_import", None)

    if quote is None:
        print("No pending custom import quote.")
        return None

    rock = quote["rock"]
    cost = quote["cost"]

    if game.money < cost:
        print(f"Not enough money. Need ${cost}, have ${game.money}.")
        return None

    # Assign current next_id at purchase time to avoid ID collisions.
    rock.id = game.next_id
    game.next_id += 1

    rock.parents = None
    rock.generation = game.generation
    rock.imported = True
    rock.sold = False

    game.money -= cost
    game.rocks[rock.id] = rock

    evaluate_rock_value(rock)
    activate_game(game)

    game.events.append(
        f"Bought custom imported rock #{rock.id} for ${cost}."
    )

    game.pending_custom_import = None

    print(f"Bought custom imported rock #{rock.id} {rock.name} for ${cost}.")
    print(f"Cash money is now ${game.money}.")

    return rock

def build_requested_gene_overrides_from_widgets(request_widgets):
    """
    Convert checkbox/dropdown widgets into requested gene values.

    Dependency-safe:
    - eye_color ignored unless eyes are requested
    - hair_color ignored unless hair is requested
    - hair_texture ignored unless hair is requested
    """
    requested_values = {}

    for gene_name, controls in request_widgets.items():
        checkbox = controls["checkbox"]
        dropdown = controls["dropdown"]

        if not checkbox.value:
            continue

        if not requested_trait_dependency_satisfied(gene_name, request_widgets):
            continue

        requested_values[gene_name] = str(dropdown.value)

    return requested_values


# ============================================================
# REQUESTED IMPORT: HIDDEN RECESSIVE ROLL RULES
# ============================================================

def get_max_requested_trait_level(gene_name):
    """
    Find the highest nonzero allele level listed in REQUEST_TRAIT_CATALOG
    for a given trait.

    Example:
        eye_color options include 00, 11, 22, 33, 44, 55
        -> max_level = 5
    """
    if gene_name not in REQUEST_TRAIT_CATALOG:
        return None

    max_level = 0

    for gene_value in REQUEST_TRAIT_CATALOG[gene_name]["options"].values():
        gene_value = str(gene_value)

        for ch in gene_value:
            if ch.isdigit():
                max_level = max(max_level, int(ch))

    return max_level

REQUEST_RECESSIVE_ROLL_TRAITS = {
    "eye_color": {
        "max_level": get_max_requested_trait_level("eye_color")
    },

    "hair_color": {
        "max_level": get_max_requested_trait_level("hair_color"),
        "exact_values": {"00", "01", "11"}  # white and silver stay exact
    },
}

def requested_dependency_satisfied_from_values(gene_name, requested_values):
    """
    Backend dependency check using requested_values dictionary.

    Examples:
    - eye_color requires eyes
    - hair_color requires hair OR facial_hair OR brows
    - hair_texture requires hair OR facial_hair
    """
    dependency = REQUEST_TRAIT_DEPENDENCIES.get(gene_name, None)

    if dependency is None:
        return True

    mode = dependency.get("mode", "any")
    parent_rules = dependency.get("parents", [])

    checks = []

    for rule in parent_rules:
        parent_gene = rule["gene"]
        blocked_values = set(str(v) for v in rule.get("blocked_values", set()))

        if parent_gene not in requested_values:
            checks.append(False)
            continue

        parent_value = str(requested_values[parent_gene])

        if parent_value in blocked_values:
            checks.append(False)
            continue

        checks.append(True)

    if mode == "all":
        return all(checks)

    return any(checks)

# ============================================================
# REQUESTED IMPORT: HIDDEN RECESSIVE GENOTYPE ROLL RULES
# ============================================================

def get_catalog_alleles(gene_name, include_zero=False):
    """
    Extract allele characters from REQUEST_TRAIT_CATALOG.

    Example:
        eye_color values 00, 11, 22, 33, 44, 55
        -> ["1", "2", "3", "4", "5"] if include_zero=False
    """
    alleles = set()

    if gene_name not in REQUEST_TRAIT_CATALOG:
        return []

    for gene_value in REQUEST_TRAIT_CATALOG[gene_name]["options"].values():
        gene_value = str(gene_value)

        for ch in gene_value:
            if not ch.isdigit():
                continue

            if ch == "0" and not include_zero:
                continue

            alleles.add(ch)

    return sorted(alleles, key=lambda x: int(x))

def build_ladder_roll_rules(gene_name, include_zero=False, zero_is_no_trait=True):
    """
    Build hidden-recessive rules for simple dominance-ladder traits.

    If allele order is:
        1 > 2 > 3 > 4 > 5

    Then:
        11 -> 11, 12, 13, 14, 15
        22 -> 22, 23, 24, 25
        55 -> 55

    zero_is_no_trait=True means:
        00 stays exactly 00.
    """
    alleles = get_catalog_alleles(gene_name, include_zero=include_zero)

    rules = {}

    if zero_is_no_trait:
        rules["00"] = ["00"]

    for i, allele in enumerate(alleles):
        if allele == "0" and zero_is_no_trait:
            continue

        selected_gene = f"{allele}{allele}"
        hidden_options = alleles[i:]

        rules[selected_gene] = [
            f"{allele}{hidden}"
            for hidden in hidden_options
        ]

    return rules

def symmetric_values(values):
    """
    Add reversed versions of two-allele values.

    Example:
        ["34"] -> ["34", "43"]
    """
    out = set()

    for value in values:
        value = str(value)
        out.add(value)

        if len(value) == 2:
            out.add(value[::-1])

    return sorted(out)

BODY_COLOR_ROLL_RULES = {
    # white = black codominance gives silver, so white cannot hide black
    "00": ["00", "02", "03", "04", "05", "06"],

    # black cannot hide white or it becomes silver
    "11": ["11", "12", "13", "14", "15", "16"],

    # silver must be white + black
    "01": ["01", "10"],

    # brown dominates red/yellow/blue/patchwork
    "22": ["22", "23", "24", "25", "26"],

    # red/yellow/blue mix with each other, so they can only hide patchwork
    "33": ["33", "36"],  # red
    "44": ["44", "46"],  # yellow
    "55": ["55", "56"],  # blue

    # mixed colors must stay mixed
    "34": ["34", "43"],  # orange
    "35": ["35", "53"],  # purple
    "45": ["45", "54"],  # green

    # patchwork is recessive
    "66": ["66"],
}

HAIR_COLOR_ROLL_RULES = {
    # white = black codominance gives silver, so white cannot hide black
    "00": ["00", "02", "03", "04", "05", "06"],

    # black cannot hide white or it becomes silver
    "11": ["11", "12", "13", "14", "15", "16"],

    # silver must be white + black
    "01": ["01", "10"],

    # other hair colors behave like a recessive ladder beneath white/black
    "22": ["22", "23", "24", "25", "26"],  # brown
    "33": ["33", "34", "35", "36"],        # blonde
    "44": ["44", "45", "46"],              # red
    "55": ["55", "56"],                    # pink
    "66": ["66"],                          # blue
}

DOSAGE_ROLL_RULES = {
    # one active allele vs two active alleles matters
    "eyes": {
        "00": ["00"],
        "01": ["01", "10"],
        "11": ["11"],
    },

    "hair": {
        "00": ["00"],
        "01": ["01", "10"],
        "11": ["11"],
    },

    "fuzz": {
        "00": ["00"],
        "01": ["01", "10"],
        "11": ["11"],
    },

    # arms are codominant-ish combinations
    "arms": {
        "00": ["00"],
        "01": ["01", "10"],
        "11": ["11"],
        "02": ["02", "20"],
        "12": ["12", "21"],
        "22": ["22"],
    },

    # splitting must stay exact, otherwise mitosion/spore changes
    "splitting": {
        "00": ["00"],
        "11": ["11"],
        "22": ["22"],
    },

    # gender should stay exact
    "gender": {
        "00": ["00"],
        "01": ["01", "10"],
    },

    # texture is currently exact
    "hair_texture": {
        "00": ["00"],
        "11": ["11"],
    },
}

REQUEST_GENOTYPE_ROLL_RULES = {}

# -----------------------------
# Shape/size are always visible traits.
# 00 is not "none" here, so 00 can hide higher/recessive levels.
# -----------------------------
REQUEST_GENOTYPE_ROLL_RULES["shape"] = build_ladder_roll_rules(
    "shape",
    include_zero=True,
    zero_is_no_trait=False
)

REQUEST_GENOTYPE_ROLL_RULES["size"] = build_ladder_roll_rules(
    "size",
    include_zero=True,
    zero_is_no_trait=False
)

# -----------------------------
# Special colors
# -----------------------------
REQUEST_GENOTYPE_ROLL_RULES["color"] = BODY_COLOR_ROLL_RULES
REQUEST_GENOTYPE_ROLL_RULES["hair_color"] = HAIR_COLOR_ROLL_RULES

# -----------------------------
# Simple dominance-ladder traits.
# 00 means none/no trait, so 00 stays exact.
# -----------------------------
for gene_name in [
    "eye_color",
    "brows",
    "mouths",
    "noses",
    "crowns",
    "ears",
    "facial_hair",
]:
    if gene_name in REQUEST_TRAIT_CATALOG:
        REQUEST_GENOTYPE_ROLL_RULES[gene_name] = build_ladder_roll_rules(
            gene_name,
            include_zero=False,
            zero_is_no_trait=True
        )

# -----------------------------
# Binary exact traits.
# If requested, they stay exact for now.
# -----------------------------
for gene_name in [
    "wings",
    "halos",
    "horns",
    "wrinkles",
    "freckles",
    "stones",
    "tails",
]:
    REQUEST_GENOTYPE_ROLL_RULES[gene_name] = {
        "00": ["00"],
        "11": ["11"],
    }

# -----------------------------
# Dosage/codominant/special traits.
# -----------------------------
for gene_name, rules in DOSAGE_ROLL_RULES.items():
    REQUEST_GENOTYPE_ROLL_RULES[gene_name] = rules


# Backward-compatible alias if any old code references this name.
REQUEST_RECESSIVE_ROLL_TRAITS = REQUEST_GENOTYPE_ROLL_RULES

def make_requested_gene_value(gene_name, selected_gene_value):
    """
    Convert a requested dropdown value into the actual imported gene.

    Uses REQUEST_GENOTYPE_ROLL_RULES when available.

    Example:
        eye_color requested 22 -> randomly 22, 23, 24, or 25
        eye_color requested 55 -> 55
        hair requested 01 -> 01 or 10
        body orange requested 34 -> 34 or 43
    """
    selected_gene_value = str(selected_gene_value)

    rules = REQUEST_GENOTYPE_ROLL_RULES.get(gene_name, None)

    if rules is None:
        return selected_gene_value

    possible_values = rules.get(selected_gene_value, None)

    if not possible_values:
        return selected_gene_value

    return random.choice(possible_values)





def calculate_requested_import_cost(rock):
    """
    Requested import cost:
        max($8, ceil(2 × base value))
    """
    evaluate_rock_value(rock)

    return max(
        2,
        math.ceil(REQUESTED_IMPORT_MULTIPLIER * rock.base_value)
    )

def preview_requested_import_rock(
    game,
    requested_values,
    reroll_if_craisen=True,
    max_attempts=100
):
    """
    Generate a requested import preview.

    The rock is not added to the game yet.
    It is stored as game.pending_requested_import.
    """
    last_rock = None

    for attempt in range(1, max_attempts + 1):
        rock = make_random_rock(game.next_id)

        rock.parents = None
        rock.generation = game.generation
        rock.name = f"Requested_{rock.name}"

        apply_requested_import_policy(
            rock,
            requested_values=requested_values
        )

        ensure_rock_game_attributes(
            rock,
            imported=True,
            sold=False
        )

        evaluate_rock_value(rock)

        last_rock = rock

        if not reroll_if_craisen:
            break

        if getattr(rock, "is_craisen", 0) != 1:
            break

    cost = calculate_requested_import_cost(last_rock)

    quote = {
        "rock": last_rock,
        "cost": cost,
        "requested_values": dict(requested_values),
        "attempts": attempt,
        "reroll_if_craisen": reroll_if_craisen,
    }

    game.pending_requested_import = quote

    print("====================================")
    print("REQUESTED IMPORT QUOTE")
    print("====================================")
    print(f"Rock: #{last_rock.id} {last_rock.name}")
    print(f"Generation: {last_rock.generation}")
    print(f"Base value: ${last_rock.base_value}")
    print(f"Requested import cost: ${cost}")
    print(f"Craisen: {bool(last_rock.is_craisen)}")
    print(f"Attempts: {attempt}")
    print("------------------------------------")
    print("Requested traits:")

    if len(requested_values) == 0:
        print("- None selected")
    else:
        for gene_name, gene_value in requested_values.items():
            label = REQUEST_TRAIT_CATALOG.get(gene_name, {}).get("label", gene_name)
            print(f"- {label}: {gene_value}")

    print("====================================")

    return quote

def buy_requested_import_quote(game, quote=None):
    """
    Buy the pending requested import.
    """
    if quote is None:
        quote = getattr(game, "pending_requested_import", None)

    if quote is None:
        print("No pending requested import quote.")
        return None

    rock = quote["rock"]
    cost = quote["cost"]

    if game.money < cost:
        print(f"Not enough money. Need ${cost}, have ${game.money}.")
        return None

    rock.id = game.next_id
    game.next_id += 1

    rock.parents = None
    rock.generation = game.generation
    rock.imported = True
    rock.sold = False

    game.money -= cost
    game.rocks[rock.id] = rock

    evaluate_rock_value(rock)
    activate_game(game)

    game.events.append(
        f"Bought requested imported rock #{rock.id} for ${cost}."
    )

    game.pending_requested_import = None

    print(f"Bought requested imported rock #{rock.id} {rock.name} for ${cost}.")
    print(f"Cash money is now ${game.money}.")

    return rock

def make_requested_import_builder(game, on_refresh=None):
    """
    Build a checkbox/dropdown requested-import menu.

    Dependency rules:
    - Eye Color requires Eyes
    - Hair Color requires Hair
    - Hair Texture requires Hair
    """
    request_widgets = {}
    rows = []

    for gene_name, info in REQUEST_TRAIT_CATALOG.items():
        label = info["label"]
        options = info["options"]

        checkbox = widgets.Checkbox(
            value=False,
            description=label,
            indent=False,
            layout=widgets.Layout(
                width="210px",
                min_height="34px"
            )
        )

        dropdown = widgets.Dropdown(
            options=[(option_label, gene_value) for option_label, gene_value in options.items()],
            layout=widgets.Layout(
                width="260px",
                min_height="34px"
            )
        )

        note = widgets.HTML(
            value="",
            layout=widgets.Layout(
                width="150px",
                min_height="34px"
            )
        )

        request_widgets[gene_name] = {
            "checkbox": checkbox,
            "dropdown": dropdown,
            "note": note,
        }

        row = widgets.HBox(
            [checkbox, dropdown, note],
            layout=widgets.Layout(
                width="650px",
                min_height="42px",
                margin="0 0 8px 0",
                align_items="center"
            )
        )

        rows.append(row)

    reroll_craisen_checkbox = widgets.Checkbox(
        value=True,
        description="Reroll if craisen",
        indent=False
    )

    preview_button = widgets.Button(
        description="Preview Requested Rock",
        button_style="info"
    )

    buy_button = widgets.Button(
        description="Buy Requested Rock",
        button_style="success"
    )

    clear_button = widgets.Button(
        description="Clear Checks",
        button_style="warning"
    )

    builder_out = widgets.Output()

    scroll_box = widgets.Box(
        [widgets.VBox(rows)],
        layout=widgets.Layout(
            max_height="500px",
            overflow_y="auto",
            border="1px solid #cccccc",
            padding="12px",
            width="700px"
        )
    )

    def refresh_dependency_states(*args):
        update_requested_trait_dependency_states(request_widgets)

    # Parent dependency triggers
    for parent_gene in ["eyes", "hair"]:
        if parent_gene in request_widgets:
            request_widgets[parent_gene]["checkbox"].observe(
                refresh_dependency_states,
                names="value"
            )
            request_widgets[parent_gene]["dropdown"].observe(
                refresh_dependency_states,
                names="value"
            )

    # Initialize dependency states.
    update_requested_trait_dependency_states(request_widgets)

    def on_preview(_):
        with builder_out:
            clear_output(wait=True)

            requested_values = build_requested_gene_overrides_from_widgets(request_widgets)

            quote = preview_requested_import_rock(
                game,
                requested_values=requested_values,
                reroll_if_craisen=reroll_craisen_checkbox.value
            )

            show_rocks(
                {quote["rock"].id: quote["rock"]},
                cols=1,
                figsize_per_rock=4,
                show_traits=True,
                normalize_size=False,
                title="Requested Import Preview"
            )

    def on_buy(_):
        with builder_out:
            clear_output(wait=True)

            bought = buy_requested_import_quote(game)

            if bought is not None:
                show_rocks(
                    {bought.id: bought},
                    cols=1,
                    figsize_per_rock=4,
                    show_traits=True,
                    normalize_size=False,
                    title="Bought Requested Import"
                )

                show_game_status(game)

                if on_refresh is not None:
                    on_refresh()

    def on_clear(_):
        for controls in request_widgets.values():
            controls["checkbox"].value = False

        update_requested_trait_dependency_states(request_widgets)

        with builder_out:
            clear_output(wait=True)
            print("Requested import checks cleared.")

    preview_button.on_click(on_preview)
    buy_button.on_click(on_buy)
    clear_button.on_click(on_clear)

    ui = widgets.VBox([
        widgets.HTML("<h3>Requested Rock Import Builder</h3>"),
        widgets.HTML(
            "Check traits you want to force. Unchecked traits are suppressed, "
            "but many can still carry hidden recessive alleles."
        ),
        widgets.HTML(
            "<b>Dependency rule:</b> Eye Color requires Eyes. "
            "Hair Color and Hair Texture require Hair."
        ),
        scroll_box,
        reroll_craisen_checkbox,
        widgets.HBox([preview_button, buy_button, clear_button]),
        builder_out
    ])

    return ui

def import_random_rock(game, cost=RANDOM_IMPORT_COST, force_gender=None):
    """
    Buy/import a fully random rock for a flat cost.

    This does NOT reroll craisen.
    It is a gamble.
    """
    if game.money < cost:
        print(f"Not enough money to import rock. Need ${cost}, have ${game.money}.")
        return None

    rock_id = game.next_id
    game.next_id += 1

    rock = make_random_rock(rock_id)

    rock.generation = game.generation
    rock.parents = None
    rock.name = f"RandomImport_{rock.name}"

    apply_import_gene_overrides(
        rock,
        gene_overrides=None,
        force_gender=force_gender
    )

    ensure_rock_game_attributes(
        rock,
        imported=True,
        sold=False
    )

    game.money -= cost
    game.rocks[rock_id] = rock

    evaluate_rock_value(rock)

    game.events.append(
        f"Imported random rock #{rock.id} for ${cost}."
    )

    activate_game(game)

    print(f"Imported random rock #{rock.id} {rock.name} for ${cost}.")
    print(f"Value: ${rock.base_value} | Craisen: {bool(rock.is_craisen)}")
    print(f"Cash money is now ${game.money}.")

    return rock


def show_game_rocks(game, cols=4, figsize_per_rock=4, include_sold=False):
    """
    Show current game rocks.
    """
    evaluate_all_rocks(game)

    if include_sold:
        rock_dict = game.rocks
        title = "All Game Rocks"
    else:
        rock_dict = get_active_rocks(game)
        title = "Active Game Rocks"

    show_rocks(
        rock_dict,
        cols=cols,
        figsize_per_rock=figsize_per_rock,
        show_traits=True,
        normalize_size=False,
        title=title,
        sort_by_generation=True
    )

def show_potion_shop():
    print("====================================")
    print("POTION SHOP")
    print("====================================")

    for key, info in POTION_SHOP.items():
        print(f"{key}: {info['name']} — ${info['cost']}")
        print(f"   {info['description']}")

    print("====================================")

def show_inventory(game):
    print("====================================")
    print("INVENTORY")
    print("====================================")
    print(f"Cash money: ${game.money}")

    if len(game.potions) == 0:
        print("No potions.")
    else:
        for potion_key, count in game.potions.items():
            name = POTION_SHOP.get(potion_key, {}).get("name", potion_key)
            print(f"{name}: {count}")

    print("====================================")

def buy_potion(game, potion_key):
    if potion_key not in POTION_SHOP:
        print(f"Unknown potion: {potion_key}")
        return False

    potion = POTION_SHOP[potion_key]
    cost = potion["cost"]

    if game.money < cost:
        print(f"Not enough money. Need ${cost}, have ${game.money}.")
        return False

    game.money -= cost
    game.potions[potion_key] = game.potions.get(potion_key, 0) + 1

    game.events.append(
        f"Bought {potion['name']} for ${cost}."
    )

    print(f"Bought {potion['name']} for ${cost}.")
    print(f"Cash money is now ${game.money}.")

    return True

def show_rock_money_table(game, include_sold=False):
    evaluate_all_rocks(game)

    print("================================================================================")
    print("ROCK MONEY TABLE")
    print("================================================================================")
    print(f"{'ID':>4} {'Name':<24} {'Gen':>4} {'Base':>6} {'Sell':>6} {'Score':>6} {'Flags'}")
    print("--------------------------------------------------------------------------------")

    for rid, rock in sorted(game.rocks.items()):
        if getattr(rock, "sold", False) and not include_sold:
            continue

        flags = []

        if getattr(rock, "sold", False):
            flags.append("SOLD")
        if getattr(rock, "imported", False):
            flags.append("IMPORTED")
        if getattr(rock, "used_as_parent", False):
            flags.append("BRED_PARENT")
        if getattr(rock, "is_craisen", 0) == 1:
            flags.append("CRAISEN")

        flag_text = ", ".join(flags) if len(flags) > 0 else "OK"

        print(
            f"{rid:>4} "
            f"{rock.name:<24.24s} "
            f"{rock.generation:>4} "
            f"${rock.base_value:>5} "
            f"${rock.sell_value:>5} "
            f"${rock.score_value:>5} "
            f"{flag_text}"
        )

    print("================================================================================")

# ============================================================
# SAVE / LOAD SYSTEM
# ============================================================

import json
from datetime import datetime

SAVE_VERSION = "0.1.0"

def rock_to_dict(rock):
    """
    Convert a Rock object into a JSON-safe dictionary.

    We save core rock identity, genes, lineage, and gameplay flags.
    Runtime-only calculated values are also saved for readability,
    but are recalculated after loading.
    """
    ensure_rock_game_attributes(rock)

    parents = getattr(rock, "parents", None)
    if parents is not None:
        parents = list(parents)

    return {
        "id": int(rock.id),
        "name": str(rock.name),
        "genes": dict(rock.genes),
        "parents": parents,
        "generation": int(getattr(rock, "generation", 0)),

        # Gameplay state
        "sold": bool(getattr(rock, "sold", False)),
        "imported": bool(getattr(rock, "imported", False)),
        "used_as_parent": bool(getattr(rock, "used_as_parent", False)),
        "dead": bool(getattr(rock, "dead", False)),
        "puffed": bool(getattr(rock, "puffed", False)),
        "death_reason": getattr(rock, "death_reason", None),

        # Cached values, recalculated on load
        "base_value": int(getattr(rock, "base_value", 0)),
        "sell_value": int(getattr(rock, "sell_value", 0)),
        "score_value": int(getattr(rock, "score_value", 0)),
        "is_craisen": int(getattr(rock, "is_craisen", 0)),
        "rock_cost": int(getattr(rock, "rock_cost", 0)),

        # Gender cache
        "gender": getattr(rock, "gender", None),
    }

def rock_from_dict(data):
    """
    Rebuild a Rock object from saved dictionary data.
    """
    parents = data.get("parents", None)

    if parents is not None:
        parents = tuple(int(p) for p in parents)

    rock = Rock(
        id=int(data["id"]),
        name=str(data.get("name", f"Rock_{data['id']}")),
        genes=dict(data.get("genes", {})),
        parents=parents,
        generation=int(data.get("generation", 0))
    )

    ensure_rock_game_attributes(rock)

    # Restore gameplay state
    rock.sold = bool(data.get("sold", False))
    rock.imported = bool(data.get("imported", False))
    rock.used_as_parent = bool(data.get("used_as_parent", False))
    rock.dead = bool(data.get("dead", False))
    rock.puffed = bool(data.get("puffed", False))
    rock.death_reason = data.get("death_reason", None)

    # Restore cached values, then recalculate later
    rock.base_value = int(data.get("base_value", 0))
    rock.sell_value = int(data.get("sell_value", 0))
    rock.score_value = int(data.get("score_value", 0))
    rock.is_craisen = int(data.get("is_craisen", 0))
    rock.rock_cost = int(data.get("rock_cost", 0))

    # Restore/sync gender
    if data.get("gender", None) is not None:
        rock.gender = data.get("gender", None)

    if "gender" in rock.genes:
        try:
            rock.gender = express_gender_from_gene(rock.genes["gender"])
        except Exception:
            pass

    return rock

def serialize_breeding_queue_entry(entry):
    """
    Convert breeding queue entry to JSON-safe format.

    Supports:
    - old tuple format: (a, b)
    - new dict format: {"parents": (a, b), "potion": potion_key}
    """
    if isinstance(entry, dict):
        a, b = get_queue_entry_pair(entry)
        potion = get_queue_entry_potion(entry)

        return {
            "parents": [int(a), int(b)],
            "potion": potion
        }

    a, b = entry

    return {
        "parents": [int(a), int(b)],
        "potion": None
    }

def deserialize_breeding_queue_entry(data):
    """
    Rebuild breeding queue entry from JSON-safe format.
    """
    parents = data.get("parents", None)

    if parents is None or len(parents) != 2:
        return None

    potion = data.get("potion", None)

    return {
        "parents": (int(parents[0]), int(parents[1])),
        "potion": potion
    }

def game_to_dict(game):
    """
    Convert full GameState into a JSON-safe dictionary.
    """
    evaluate_all_rocks(game)

    save_data = {
        "save_version": SAVE_VERSION,
        "saved_at": datetime.now().isoformat(timespec="seconds"),

        "game": {
            "player_name": getattr(game, "player_name", ""),
            "cabal_curse_enabled": bool(getattr(game, "cabal_curse_enabled", False)),
            "market_pod_used_generations": list(getattr(game, "market_pod_used_generations", [])),
            "generation": int(game.generation),
            "max_generation": int(game.max_generation),
            "money": int(game.money),
            "next_id": int(game.next_id),
            "max_pairs_per_generation": int(game.max_pairs_per_generation),
            "game_over": bool(game.game_over),

            "potions": dict(game.potions),
            "events": list(game.events),

            "breeding_queue": [
                serialize_breeding_queue_entry(entry)
                for entry in game.breeding_queue
            ],

            "rocks": [
                rock_to_dict(rock)
                for _, rock in sorted(game.rocks.items())
            ],
        }
    }

    return save_data

def game_from_dict(save_data):
    """
    Rebuild GameState from a saved dictionary.
    """
    if "game" not in save_data:
        raise ValueError("Invalid save file: missing 'game' field.")

    g = save_data["game"]

    game = GameState(
        rocks={},
        next_id=int(g.get("next_id", 1)),
        generation=int(g.get("generation", 0)),
        max_generation=int(g.get("max_generation", DEFAULT_MAX_GENERATION)),
        money=int(g.get("money", DEFAULT_STARTING_MONEY)),
        max_pairs_per_generation=int(
            g.get("max_pairs_per_generation", DEFAULT_MAX_PAIRS_PER_GENERATION)
        ),
        breeding_queue=[],
        potions=dict(g.get("potions", {})),
        events=list(g.get("events", [])),
        game_over=bool(g.get("game_over", False))
    )

    #game.player_name = data.get("player_name", "")
    #game.cabal_curse_enabled = bool(data.get("cabal_curse_enabled", False))
    #game.market_pod_used_generations = data.get("market_pod_used_generations", [])

    ensure_market_state(game)
    ensure_player_profile_state(game)

    # Rebuild rocks
    for rock_data in g.get("rocks", []):
        rock = rock_from_dict(rock_data)
        game.rocks[int(rock.id)] = rock

    # Rebuild queue
    queue = []

    for entry_data in g.get("breeding_queue", []):
        entry = deserialize_breeding_queue_entry(entry_data)
        if entry is not None:
            queue.append(entry)

    game.breeding_queue = queue

    # Safety: ensure next_id is above all rock IDs
    if len(game.rocks) > 0:
        game.next_id = max(game.next_id, max(game.rocks.keys()) + 1)

    # Recalculate values and parent flags
    evaluate_all_rocks(game)

    game.events.append("Game loaded from save file.")

    return game

def game_to_json_string(game, indent=2):
    """
    Convert game to pretty JSON string.
    """
    return json.dumps(
        game_to_dict(game),
        indent=indent,
        sort_keys=False
    )

def game_from_json_string(json_string):
    """
    Load game from JSON string.
    """
    save_data = json.loads(json_string)
    return game_from_dict(save_data)

def save_game_json(game, filepath):
    """
    Save game to a JSON file path.
    Useful outside Streamlit.
    """
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(game_to_json_string(game))

    return filepath

def load_game_json(filepath):
    """
    Load game from a JSON file path.
    Useful outside Streamlit.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        json_string = f.read()

    return game_from_json_string(json_string)

def make_save_filename(game):
    """
    Create a friendly save filename.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"rock_game_gen{game.generation}_money{game.money}_{timestamp}.json"

def describe_rock_for_breeding(rock):
    """
    Compact description of one rock for breeding UI/debug.
    """
    evaluate_rock_value(rock)

    status_bits = []

    if is_rock_sold(rock):
        status_bits.append("SOLD")

    if is_rock_craisen(rock):
        status_bits.append("CRAISEN")

    if len(status_bits) == 0:
        status_bits.append("OK")

    parent_text = "Founder/import"
    if rock.parents is not None:
        parent_text = f"Parents: #{rock.parents[0]} and #{rock.parents[1]}"

    return (
        f"#{rock.id} {rock.name} | "
        f"{get_rock_gender_name(rock)} | "
        f"Gen {rock.generation} | "
        f"Value ${rock.sell_value} | "
        f"{', '.join(status_bits)} | "
        f"{parent_text}"
    )

def preview_breeding_pair(game, parent_a_id, parent_b_id):
    """
    Print a readable breeding-pair validation report.
    """
    result = validate_breeding_pair(game, parent_a_id, parent_b_id)

    parent_a = result["parent_a"]
    parent_b = result["parent_b"]

    print("====================================")
    print("BREEDING PAIR PREVIEW")
    print("====================================")

    if parent_a is not None:
        print("Parent A:", describe_rock_for_breeding(parent_a))
    else:
        print(f"Parent A: #{parent_a_id} not found")

    if parent_b is not None:
        print("Parent B:", describe_rock_for_breeding(parent_b))
    else:
        print(f"Parent B: #{parent_b_id} not found")

    print("------------------------------------")

    if result["valid"]:
        print("Status: VALID PAIR")
    else:
        print("Status: INVALID PAIR")

    if len(result["errors"]) > 0:
        print("\nErrors:")
        for error in result["errors"]:
            print(f"- {error}")

    if len(result["warnings"]) > 0:
        print("\nWarnings:")
        for warning in result["warnings"]:
            print(f"- {warning}")

    print("====================================")

    return result




























