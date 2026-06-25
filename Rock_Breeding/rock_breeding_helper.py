#-----------------------------------------------------
"""
Rock Breeding Helper 

This file answers:

- How does a rock breed?????????
- Does interbreeding hurt my rock? - YES -
    - How much interbreeding is this rock pair since I have so many?

- Can these rocks can be used as parents?
- My rock child died at birth! :(
- My rock child clutch was massive, or did they duplicate?

"""
#-----------------------------------------------------

#-----------------------------------------------------
# IMPORT ZONE
#-----------------------------------------------------

import random, math

from dataclasses import dataclass, field
from typing import List

#-----------------------------------------------------
# SPECIAL IMPORT ZONE
#-----------------------------------------------------

import Rock_Genetics.rock_genetic_helper as genetics

"""
To run module individually: python -m Rock_Breeding.rock_breeding_helper
"""

#-----------------------------------------------------
# ROCK Name ZONE
#-----------------------------------------------------

NAME_BITS_START = [
    "Grum", "Peb", "Bas", "Quartz", "Moss", "Igni", "Crag", "Glim", "Obsi", "Feld"
]

NAME_BITS_END = [
    "ble", "ite", "or", "yx", "stone", "ling", "rock", "spar", "gem", "oid"
]

#-----------------------------------------------------
# ROCK BASE GAME NUMBERS
#-----------------------------------------------------

CHILD_DEATH_CHANCE = 0.05
CLUTCH_MEAN = 1.5
CLUTCH_STD = 2.0
MAX_CLUTCH_SIZE = None

MUTATION_CHANCE = 0.01

SPORE_DEATH_CHANCE = 0.25
SPORE_CLONE_COUNT = 3

#-----------------------------------------------------
# BREEDING ORCHESTRATOR DEFINITION ZONE
#-----------------------------------------------------

@dataclass()
class BreedingMaster: 

    name_bits_start = NAME_BITS_START
    name_bits_end = NAME_BITS_END

    child_death_chance = CHILD_DEATH_CHANCE

    spore_death_chance = SPORE_DEATH_CHANCE
    spore_clone_count = SPORE_CLONE_COUNT

    clutch_mean = CLUTCH_MEAN
    clutch_std = CLUTCH_STD
    max_clutch_size = MAX_CLUTCH_SIZE

    child_gene_mutation_chance = MUTATION_CHANCE

    GenomeFactory: genetics.GenomeFactory = field(default_factory = genetics.GenomeFactory)

    ExpressionEngine: genetics.ExpressionEngine = field(default_factory = genetics.ExpressionEngine)

    ValueCalculator: genetics.ValueCalculator = field(default_factory = genetics.ValueCalculator)

    child_bred_in_generation: list[genetics.Rock] = field(default_factory = list)

    #-----------------------------------------------------
    # GETTING ROCK NAME
    #-----------------------------------------------------

    def random_rock_name(
        self,
    ) -> str:
        return random.choice(self.name_bits_start) + random.choice(self.name_bits_end)

    #-----------------------------------------------------
    # BREEDING ONE CHILD, SET VALUE, SET ACTIVE
    #-----------------------------------------------------

    def breed_child_from_parents(
        self,
        parent_a: genetics.Rock,
        parent_b: genetics.Rock,

        next_id,
        child_generation,

        mutation_chance = None,

    ):
        """
        Breed one child from a valid parent pair.

        Does not advance generation by itself.
        """

        child_id = next_id
        child_name = self.random_rock_name()

        if random.random() < 0.5:
            child_sex = genetics.Sex.MALE
        else:
            child_sex = genetics.Sex.FEMALE

        if mutation_chance == None:
            mutation_chance = self.child_gene_mutation_chance

        child_genes = self.GenomeFactory.make_child_rock_genome_from_parents(
            parent_a = parent_a,
            parent_b = parent_b,
            mutation_chance = mutation_chance,
        )

        child_death_genes = self.GenomeFactory.inherit_death_genes(
            parent_a = parent_a,
            parent_b = parent_b,
            mutation_chance = mutation_chance,
        )

        child = genetics.Rock(
            id = child_id,
            name = child_name,
            sex = child_sex,

            genotype = child_genes,
            death_genes = child_death_genes,
            parent_ids = [parent_a.id, parent_b.id],
            generation = child_generation,
            status = genetics.RockStatus.ACTIVE
        )

        child = self.ExpressionEngine.instantiate_phenotype(
            rock = child,
        )

        child = self.ValueCalculator.set_rock_value(
            rock = child,
        )

        return (child)

    #-----------------------------------------------------
    # GETTING NECESSARIES FOR A FULL ROCK CLUTCH
    #-----------------------------------------------------
    
    def validate_breeding_pair(
        self,
        parent_a: genetics.Rock,
        parent_b: genetics.Rock,
        require_opposite_gender = True,
    ):
        """
        Validate whether two rocks can breed.
        """

        errors = []
        warnings = [] 

        if parent_a is None or parent_b is None:
            return {
                "valid": False,
                "errors": errors,
                "warnings": warnings,
                "parent_a": parent_a,
                "parent_b": parent_b,
            }

        if parent_a.id == parent_b.id:
            errors.append("A rock cannot breed with itself.")

        if not parent_a.is_active:
            errors.append(f"parent a is not breedable, but is: {parent_a.status}")

        if not parent_b.is_active:
            errors.append(f"parent b is not breedable, but is: {parent_b.status}")

        if require_opposite_gender and parent_a.sex == parent_b.sex:
            errors.append(
                f"Parents must be opposite gender. "
                f"Rock #{parent_a.id} is {parent_a.sex} and "
                f"Rock #{parent_b.id} is {parent_b.sex}."
            )

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "parent_a": parent_a,
            "parent_b": parent_b,
        }

    def roll_clutch_size(
            self,
            mean = None,
            std = None, 
            max_clutch_size = None,
    ):
        """
        Emulates Excel:
        ABS(INT(NORMINV(RAND(), 1.5, 2))) + 1

        In Python:
        NORMINV(RAND(), mean, std) is equivalent to a normal draw.
        Excel INT floors toward negative infinity, so use math.floor.
        """
        if mean == None:
            mean = self.clutch_mean

        if std == None:
            std = self.clutch_std

        if max_clutch_size == None:
            max_clutch_size = self.max_clutch_size

        x = random.gauss(mean, std)
        clutch = abs(math.floor(x)) + 1

        if max_clutch_size is not None:
            clutch = min(clutch, max_clutch_size)

        return clutch

    def maybe_mitote_child(
        self,
        child: genetics.Rock, 
    ):
        """
        Children with mitosion active split into two rocks, 
        this creates a clone without the name indicator of old!

        returns the mitoted_child Rock object!
        """

        child_mitote = None

        if child.genotype.genes["splitting"].phenotype == "mitosion" and child.has_split == False:

            mitote_name = self.random_rock_name()
            mitote_id = child.id + 1

            child_mitote = genetics.Rock(
                id = mitote_id,
                name = mitote_name,
                sex = child.sex,
                genotype = child.genotype,
                death_genes = child.death_genes,
                parent_ids = child.parent_ids,
                generation = child.generation,
                status = genetics.RockStatus.ACTIVE,

                has_split = True
            )

            child.has_split = True

            return child_mitote

        return child_mitote
    
    def maybe_spore_child(
        self,
        child: genetics.Rock, 
        spore_death_chance = None,
        spore_clone_count = None,
    ):
        """
        Children with spore active split into 4 rocks, 
        this creates the clones without the name indicator of old,
        and computes the risky spore death chance of SPORE_DEATH

        returns the child_spore Rock list: only contains child if no sporing!!
        """
        if spore_death_chance == None:
            spore_death_chance = self.spore_death_chance

        if spore_clone_count == None:
            spore_clone_count = self.spore_clone_count

        child_spore: list[genetics.Rock] = [child]

        if child.genotype.genes["splitting"].phenotype == "spore" and child.has_split == False:
            for spore_clone in range(1 + spore_clone_count):
                
                spore_name = self.random_rock_name()
                spore_id = child.id + 1 + spore_clone

                child_puff = genetics.Rock(
                    id = spore_id,
                    name = spore_name,
                    sex = child.sex,
                    genotype = child.genotype,
                    death_genes = child.death_genes,
                    parent_ids = child.parent_ids,
                    generation = child.generation,
                    status = genetics.RockStatus.ACTIVE,

                    has_split = True
                )

                if random.random() < spore_death_chance:
                    child_puff.change_status(new_status = genetics.RockStatus.DEAD)
                    child_puff.death_reason = "puffed out at birth"
                
                child_spore.append(child_puff)

            child.has_split = True

            return child_spore

        return child_spore
    
    def maybe_kill_child(
        self,
        child: genetics.Rock, 
        death_chance = None,
    ):
        """
        Child has a percent chance of dying after birth.
        Dead children remain in the tree but are worthless and cannot breed.
        """
        if death_chance == None:
            death_chance = self.child_death_chance

        if random.random() < death_chance:
            child.change_status(new_status = genetics.RockStatus.DEAD)
            child.death_reason = "died after birth"
            return child

        return child
    
    def keep_ids_on_track(
        self,
    ):
        
        pass
    #-----------------------------------------------------
    # BREEDING FOR A FULL ROCK CLUTCH
    #-----------------------------------------------------
    
    """
    result = self.validate_breeding_pair(
            parent_a = parent_a, 
            parent_b = parent_b
        )

        if not result["valid"]:
            raise ValueError("Invalid breeding pair: " + "; ".join(result["errors"]))

        parent_a = result["parent_a"]
        parent_b = result["parent_b"]
    """
#-----------------------------------------------------
# THESE WILL PROBABLY GO INTO GAMEMASTER
#-----------------------------------------------------
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

def apply_reroll_to_clutch(clutch_size, potion_key):
    """
    Take the best of two rolls for the clutch.
    """
    potion_key = normalize_potion_key(potion_key)

    if potion_key == "reroll":
        return max(clutch_size, roll_clutch_size())

    return clutch_size


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

# ============================================================
# BREEDABLE CHECKER
# ============================================================

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

# ============================================================
# LATER BREEDING QUEUE CHECKER CODE...
# ============================================================

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






















