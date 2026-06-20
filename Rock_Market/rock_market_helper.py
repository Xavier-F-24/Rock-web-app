#-----------------------------------------------------
"""
Rock Market Helper 

This file answers:

- How does one sell a rock for money?
- How do I buy a potion, what do they do?

- I want to import a random rock?
- I want to participate in the rock breeding market!

"""
#-----------------------------------------------------

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

# ============================================================
# BREEDING POD MARKET + ID
# ============================================================

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

# ============================================================
# SELLING YOUR CHILDREN
# ============================================================

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

# ============================================================
# IMPORTING SLEEK NEW SEXY MODELS
# ============================================================

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

# -----------------------------
# POTION SHOP
# -----------------------------

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



