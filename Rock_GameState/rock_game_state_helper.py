#-----------------------------------------------------
"""
Rock Game State Helper 




"""
#-----------------------------------------------------
# IMPORTANT VALUES FOR GAMEPLAY
#-----------------------------------------------------

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

#for GameState
"""
    def check_craisen(self):
        for death_gene in self.death_genes.genes:
            if self.death_genes.genes[death_gene].allele_a.value == self.death_genes.genes[death_gene].allele_b.value:
                if random.random() < CRAISEN_CHANCES:
                    self.change_status(RockStatus.CRAISENED)

"""

from functools import wraps


def trace(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        print(f"Calling {func.__name__}")
        result = func(*args, **kwargs)
        print(f"Finished {func.__name__}")
        return result

    return wrapper


def requires_money(cost):
    def decorator(func):
        @wraps(func)
        def wrapper(game, *args, **kwargs):
            if game.money < cost:
                raise ValueError(f"Not enough money. Need ${cost}, have ${game.money}.")

            game.money -= cost
            return func(game, *args, **kwargs)

        return wrapper

    return decorator












