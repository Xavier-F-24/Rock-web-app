#-----------------------------------------------------
"""
Rock GAME Helper 

This file answers:

- How does the game run?
- What are the rules of the game?

- What data needs to be pushed to run a game?

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

# ============================================================
# GAME AND GAMESTATE GROUNDWORK
# ============================================================

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

# ============================================================
# BELONGS IN GAME?
# ============================================================

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












