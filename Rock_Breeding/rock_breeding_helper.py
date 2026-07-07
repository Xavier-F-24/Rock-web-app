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
# ROCK BASE GAME NUMBERS
#-----------------------------------------------------

CHILD_DEATH_CHANCE = 0.05
CRAISEN_DEATH_CHANCE = 0.50

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

    child_death_chance = CHILD_DEATH_CHANCE
    craisen_death_chance = CRAISEN_DEATH_CHANCE

    spore_death_chance = SPORE_DEATH_CHANCE
    spore_clone_count = SPORE_CLONE_COUNT

    clutch_mean = CLUTCH_MEAN
    clutch_std = CLUTCH_STD
    max_clutch_size = MAX_CLUTCH_SIZE

    child_gene_mutation_chance = MUTATION_CHANCE

    GenomeFactory: genetics.GenomeFactory = field(default_factory = genetics.GenomeFactory)

    ExpressionEngine: genetics.ExpressionEngine = field(default_factory = genetics.ExpressionEngine)

    ValueCalculator: genetics.ValueCalculator = field(default_factory = genetics.ValueCalculator)

    NameGenerator: genetics.NameGenerator = field(default_factory = genetics.NameGenerator)

    child_bred_for_parents: list[genetics.Rock] = field(default_factory = list)

    rng: random.Random = field(default_factory = random.Random)

    #-----------------------------------------------------
    # GETTING ROCK NAME
    #-----------------------------------------------------

    @staticmethod
    def default(
        value, 
        fallback
    ):
        return fallback if value is None else value

    def random_rock_name(
        self,
        sex: genetics.Sex,
        parent_a: genetics.Rock | None = None,
        parent_b: genetics.Rock | None = None,

        force_family: bool = False,
        force_honorific: bool = False,
        force_epithet: bool = False,

    ) -> genetics.RockName:
        
        name = self.NameGenerator.generate_name(
            sex = sex,
            parent_a = parent_a,
            parent_b = parent_b,

            force_family = False,
            force_honorific = False,
            force_epithet = False,
        )

        return (name)

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

        if self.rng.random() < 0.5:
            child_sex = genetics.Sex.MALE
        else:
            child_sex = genetics.Sex.FEMALE
        
        child_name = self.NameGenerator.generate_name(
            sex = child_sex,
            parent_a = parent_a,
            parent_b = parent_b,

            force_family = False,
            force_honorific = False,
            force_epithet = False,
        )

        mutation_chance = self.default(mutation_chance, self.child_gene_mutation_chance)

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

        return (child)

    #-----------------------------------------------------
    # GETTING NECESSARIES FOR A FULL ROCK CLUTCH
    #-----------------------------------------------------
    
    def validate_breeding_pair(
        self,
        parent_a: genetics.Rock,
        parent_b: genetics.Rock,
        require_opposite_gender = True,
        game = None,
        warn_relatedness = True,
        disallow_related = None,
    ):
        """
        Validate whether two rocks can breed.
        """

        if disallow_related is not None:
            warn_relatedness = bool(disallow_related)

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

        if game is not None and warn_relatedness:
            relationship = self.describe_relationship(game, parent_a, parent_b)
            if relationship is not None:
                warnings.append(f"Parents are related: {relationship}.")

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "parent_a": parent_a,
            "parent_b": parent_b,
        }

    @staticmethod
    def get_parent_ids(rock: genetics.Rock) -> list[int]:
        return [
            int(parent_id)
            for parent_id in getattr(rock, "parent_ids", []) or []
            if parent_id is not None
        ]

    def get_ancestors(
        self,
        game,
        rock: genetics.Rock,
        max_depth: int = 12,
    ) -> set[int]:
        ancestors: set[int] = set()

        def walk(current, depth):
            if current is None or depth > max_depth:
                return

            for parent_id in self.get_parent_ids(current):
                if parent_id in ancestors:
                    continue

                ancestors.add(parent_id)
                parent = self.get_game_rock(game, parent_id)
                walk(parent, depth + 1)

        walk(rock, 0)
        return ancestors

    def get_ancestor_path_distances(
        self,
        game,
        rock: genetics.Rock,
        include_self: bool = False,
        max_depth: int = 12,
    ) -> dict[int, list[int]]:
        distances: dict[int, list[int]] = {}
        stack = [(rock, 0, {int(rock.id)})]

        while stack:
            current, depth, seen = stack.pop()
            if current is None or depth > max_depth:
                continue

            if include_self or depth > 0:
                distances.setdefault(int(current.id), []).append(depth)

            if depth == max_depth:
                continue

            for parent_id in self.get_parent_ids(current):
                if parent_id in seen:
                    continue

                parent = self.get_game_rock(game, parent_id)
                stack.append((parent, depth + 1, seen | {parent_id}))

        return distances

    def calculate_relatedness(
        self,
        game,
        parent_a: genetics.Rock,
        parent_b: genetics.Rock,
        max_depth: int = 12,
    ) -> tuple[float, dict[int, float]]:
        if parent_a.id == parent_b.id:
            return 1.0, {int(parent_a.id): 1.0}

        ancestors_a = self.get_ancestor_path_distances(game, parent_a, include_self=True, max_depth=max_depth)
        ancestors_b = self.get_ancestor_path_distances(game, parent_b, include_self=True, max_depth=max_depth)
        shared_ancestors = set(ancestors_a) & set(ancestors_b)

        relatedness = 0.0
        contributions = {}
        for ancestor_id in shared_ancestors:
            subtotal = 0.0
            for distance_a in ancestors_a[ancestor_id]:
                for distance_b in ancestors_b[ancestor_id]:
                    subtotal += 0.5 ** (distance_a + distance_b)

            if subtotal > 0:
                contributions[ancestor_id] = subtotal
                relatedness += subtotal

        return min(relatedness, 1.0), contributions

    @staticmethod
    def ordinal_word(value: int) -> str:
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
        return words.get(int(value), f"{value}th")

    @staticmethod
    def removal_word(value: int) -> str:
        words = {
            1: "once removed",
            2: "twice removed",
            3: "three times removed",
            4: "four times removed",
        }
        return words.get(int(value), f"{value} times removed")

    @staticmethod
    def great_prefix(count: int) -> str:
        if count <= 0:
            return ""
        if count == 1:
            return "great-"
        return f"{count}x-great-"

    @staticmethod
    def parent_term(rock: genetics.Rock) -> str:
        return "father" if rock.sex == genetics.Sex.MALE else "mother"

    @staticmethod
    def child_term(rock: genetics.Rock) -> str:
        return "son" if rock.sex == genetics.Sex.MALE else "daughter"

    @staticmethod
    def sibling_term(rock: genetics.Rock) -> str:
        return "brother" if rock.sex == genetics.Sex.MALE else "sister"

    @staticmethod
    def aunt_uncle_term(rock: genetics.Rock) -> str:
        return "uncle" if rock.sex == genetics.Sex.MALE else "aunt"

    @staticmethod
    def niece_nephew_term(rock: genetics.Rock) -> str:
        return "nephew" if rock.sex == genetics.Sex.MALE else "niece"

    def describe_actual_relationship(
        self,
        game,
        parent_a: genetics.Rock,
        parent_b: genetics.Rock,
        max_depth: int = 12,
    ) -> str | None:
        if parent_a.id == parent_b.id:
            return "same rock"

        ancestors_a = self.get_ancestor_path_distances(game, parent_a, include_self=True, max_depth=max_depth)
        ancestors_b = self.get_ancestor_path_distances(game, parent_b, include_self=True, max_depth=max_depth)

        if parent_a.id in ancestors_b:
            distance = min(ancestors_b[parent_a.id])
            if distance == 1:
                return f"{self.parent_term(parent_a)} and {self.child_term(parent_b)}"
            if distance == 2:
                return "grandparent and grandchild"
            return f"{self.great_prefix(distance - 2)}grandparent and {self.great_prefix(distance - 2)}grandchild"

        if parent_b.id in ancestors_a:
            distance = min(ancestors_a[parent_b.id])
            if distance == 1:
                return f"{self.child_term(parent_a)} and {self.parent_term(parent_b)}"
            if distance == 2:
                return "grandchild and grandparent"
            return f"{self.great_prefix(distance - 2)}grandchild and {self.great_prefix(distance - 2)}grandparent"

        shared_ancestors = set(ancestors_a) & set(ancestors_b)
        if not shared_ancestors:
            return None

        best = None
        for ancestor_id in shared_ancestors:
            for distance_a in ancestors_a[ancestor_id]:
                for distance_b in ancestors_b[ancestor_id]:
                    if distance_a == 0 or distance_b == 0:
                        continue

                    score = distance_a + distance_b
                    if best is None or score < best[0]:
                        best = (score, distance_a, distance_b)

        if best is None:
            return None

        _, distance_a, distance_b = best

        if distance_a == 1 and distance_b == 1:
            common_parent_count = sum(
                1
                for ancestor_id in shared_ancestors
                if 1 in ancestors_a[ancestor_id] and 1 in ancestors_b[ancestor_id]
            )
            sibling_type = "full" if common_parent_count >= 2 else "half"
            return f"{sibling_type} {self.sibling_term(parent_a)} and {self.sibling_term(parent_b)}"

        min_distance = min(distance_a, distance_b)
        max_distance = max(distance_a, distance_b)

        if min_distance == 1:
            if distance_a < distance_b:
                older = parent_a
                younger = parent_b
            else:
                older = parent_b
                younger = parent_a

            great_count = max_distance - 2
            older_term = f"{self.great_prefix(great_count)}{self.aunt_uncle_term(older)}"
            if great_count <= 0:
                younger_term = self.niece_nephew_term(younger)
            elif great_count == 1:
                younger_term = f"grand{self.niece_nephew_term(younger)}"
            else:
                younger_term = f"{great_count - 1}x-great-grand{self.niece_nephew_term(younger)}"

            if older is parent_a:
                return f"{older_term} and {younger_term}"
            return f"{younger_term} and {older_term}"

        cousin_degree = min_distance - 1
        removals = abs(distance_a - distance_b)
        cousin_text = f"{self.ordinal_word(cousin_degree)} cousins"

        if removals > 0:
            cousin_text += f" {self.removal_word(removals)}"

        return cousin_text

    @staticmethod
    def get_game_rock(game, rock_id):
        if hasattr(game, "get_rock"):
            return game.get_rock(rock_id)

        rocks = getattr(game, "rock_list", None) or getattr(game, "rocks", {})
        return rocks.get(int(rock_id))

    def describe_relationship(
        self,
        game,
        parent_a: genetics.Rock,
        parent_b: genetics.Rock,
    ) -> str | None:
        relatedness, _ = self.calculate_relatedness(game, parent_a, parent_b)
        if relatedness <= 0:
            return None

        child_inbreeding = relatedness / 2.0
        actual = self.describe_actual_relationship(game, parent_a, parent_b) or "related"
        return f"{actual} (R={relatedness:.4f}, F={child_inbreeding:.4f})"

    def set_parents_as_bred(
        self,
        parent_a: genetics.Rock,
        parent_b: genetics.Rock,
    ):
        parent_a.change_status(genetics.RockStatus.BRED)
        parent_b.change_status(genetics.RockStatus.BRED)

        return (parent_a, parent_b)

    def roll_clutch_size(
            self,
            mean = None,
            std = None, 
            max_clutch_size = None,

            reroll = None,
            plus_one = None,
    ):
        """
        Emulates Excel:
        ABS(INT(NORMINV(RAND(), 1.5, 2))) + 1

        In Python:
        NORMINV(RAND(), mean, std) is equivalent to a normal draw.
        Excel INT floors toward negative infinity, so use math.floor.
        """
        
        mean = self.default(mean, self.clutch_mean)

        std = self.default(std, self.clutch_std)

        max_clutch_size = self.default(max_clutch_size, self.max_clutch_size)

        x = self.rng.gauss(mean, std)
        clutch = abs(math.floor(x)) + 1

        if max_clutch_size is not None:
            clutch = min(clutch, max_clutch_size)

        if reroll is not None:
            clutch_one = clutch
            clutch_two = abs(math.floor(self.rng.gauss(mean, std))) + 1

            return max(clutch_one, clutch_two)
        
        if plus_one is not None:
            clutch += 1

        return clutch

    def maybe_mitote_child(
        self,
        parent_a: genetics.Rock,
        parent_b: genetics.Rock,
        child: genetics.Rock, 
    ):
        """
        Children with mitosion active split into two rocks, 
        this creates a clone without the name indicator of old!

        returns the mitoted_child Rock object!
        """

        child_mitote = None

        if child.genotype.genes["splitting"].phenotype == "mitosion" and child.has_split == False and child.status == genetics.RockStatus.ACTIVE:

            mitote_name = self.NameGenerator.generate_name(
                sex = child.sex,
                parent_a = parent_a,
                parent_b = parent_b,

                force_family = False,
                force_honorific = False,
                force_epithet = False,
            )

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
        parent_a: genetics.Rock,
        parent_b: genetics.Rock,
        spore_death_chance = None,
        spore_clone_count = None,
    ):
        """
        Children with spore active split into 4 rocks, 
        this creates the clones without the name indicator of old,
        and computes the risky spore death chance of SPORE_DEATH

        returns the child_spore Rock list: only contains child if no sporing!!
        """
        spore_death_chance = self.default(spore_death_chance, self.spore_death_chance)

        spore_clone_count = self.default(spore_clone_count, self.spore_clone_count)

        child_spore: list[genetics.Rock] = [child]

        if child.genotype.genes["splitting"].phenotype == "spore" and child.has_split == False and child.status == genetics.RockStatus.ACTIVE:
            for spore_clone in range(1 + spore_clone_count):
                
                spore_name = self.NameGenerator.generate_name(
                    sex = child.sex,
                    parent_a = parent_a,
                    parent_b = parent_b,

                    force_family = False,
                    force_honorific = False,
                    force_epithet = False,
                )

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

                if self.rng.random() < spore_death_chance:
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
        death_chance = self.default(death_chance, self.child_death_chance)

        if self.rng.random() < death_chance and child.status == genetics.RockStatus.ACTIVE:
            child.change_status(new_status = genetics.RockStatus.DEAD)
            child.death_reason = "died after birth"
            return child

        return child
    
    def maybe_craisen_child(
        self,
        child: genetics.Rock, 
        craisen_chance = None,
    ):
        """
        Child has a percent chance of dying after birth.
        Dead children remain in the tree but are worthless and cannot breed.
        """
        craisen_chance = self.default(craisen_chance, self.craisen_death_chance)

        craisen_possible = False

        for gene in child.death_genes.genes:
            if child.death_genes.genes[gene].allele_a == child.death_genes.genes[gene].allele_b:
                craisen_possible = True

        """
        QUESTION: WE COULD CHANGE TO CRAISEN CHANCE BEING LOWER, 
        BUT MULTIPLE HITS MEANS X TIMES THE ODDS, THOUGHT!
        """

        if craisen_possible and self.rng.random() < craisen_chance and child.checked_craisen == False and child.status == genetics.RockStatus.ACTIVE:
            child.change_status(new_status = genetics.RockStatus.CRAISENED)
            child.checked_craisen = True
            child.death_reason = "craisend up, man"
            return child

        return child
    
    def get_next_id(
        self,
    ):
        """
        Helps with mitosion / sporing
        calculates the length of the current bred children
        returns length of list for next ID (will be added to "next_id" in case)
        """
        new_id = 0

        if self.child_bred_for_parents:
            new_id = 0 + len(self.child_bred_for_parents)

        return (new_id)

    #-----------------------------------------------------
    # BREEDING FOR A FULL ROCK CLUTCH
    #-----------------------------------------------------
    
    def breed_parent_set(
        self,
        parent_a: genetics.Rock,
        parent_b: genetics.Rock,

        next_id,
        child_generation,

        mutation_chance = None,

        death_chance = None,
        craisen_chance = None,
        spore_death_chance = None,
        spore_clone_count = None,
        clutch_reroll = None,
        clutch_plus_one = None,

        # MORE???

    ):
        """
        Takes parents a and b, calculates clutch,
        breeds them with mitosion, spore, kill handling
        """

        result = self.validate_breeding_pair(
                parent_a = parent_a, 
                parent_b = parent_b
            )

        if not result["valid"]:
            raise ValueError("Invalid breeding pair: " + "; ".join(result["errors"]))

        parent_a = result["parent_a"]
        parent_b = result["parent_b"]

        mutation_chance = self.default(mutation_chance, self.child_gene_mutation_chance)
        death_chance = self.default(death_chance, self.child_death_chance)
        craisen_chance = self.default(craisen_chance, self.craisen_death_chance)
        spore_death_chance = self.default(spore_death_chance, self.spore_death_chance)
        spore_clone_count = self.default(spore_clone_count, self.spore_clone_count)

        clutch = self.roll_clutch_size(
            mean = None,
            std = None,
            max_clutch_size = None,

            reroll = clutch_reroll,
            plus_one = clutch_plus_one,
        )

        mod_id = next_id + self.get_next_id()

        for child in range(clutch):
            
            mod_id = next_id + self.get_next_id()

            child = self.breed_child_from_parents(
                parent_a = parent_a,
                parent_b = parent_b,
                next_id = mod_id,
                child_generation = child_generation,
                mutation_chance = mutation_chance,
            )

            child = self.maybe_kill_child(
                child = self.maybe_craisen_child(
                    child = child,
                    craisen_chance = craisen_chance
                    ),
                death_chance = death_chance,
            )

            self.child_bred_for_parents.append(child)

            #-----------------------------------------------------
            # HANDLE MITOSION OF CHILD
            #-----------------------------------------------------

            mitote = self.maybe_mitote_child(
                child = child,
                parent_a = parent_a,
                parent_b = parent_b,
            )

            if mitote != None:

                mitote = self.maybe_kill_child(
                    child = self.maybe_craisen_child(
                        child = mitote,
                        craisen_chance = craisen_chance
                        ),
                    death_chance = death_chance,
                )
                
                self.child_bred_for_parents.append(mitote)
 
            #-----------------------------------------------------
            # HANDLE SPORING OF CHILD
            #-----------------------------------------------------

            spore = self.maybe_spore_child(
                child = child,
                parent_a = parent_a,
                parent_b = parent_b,

                spore_death_chance = spore_death_chance,
                spore_clone_count = spore_clone_count,
            )

            if len(spore) > 1:

                spore.pop(0) # REMOVE THE ORIGINAL CHILD - ALWAYS FIRST!

                for spore_bro in spore:

                    spore_bro = self.maybe_kill_child(
                        child = self.maybe_craisen_child(
                            child = spore_bro,
                            craisen_chance = craisen_chance
                            ),
                        death_chance = death_chance,
                    )

                    self.child_bred_for_parents.append(spore_bro)

        #-----------------------------------------------------
        # MARK PARENTS AS BRED
        #-----------------------------------------------------

        (parent_a, parent_b) = self.set_parents_as_bred(
                                parent_a = parent_a,
                                parent_b = parent_b,
                            )

        #-----------------------------------------------------
        # CALCULATE CHILDREN VALUES AND PHENOTYPES
        #-----------------------------------------------------

        for child in self.child_bred_for_parents:
            child = self.ValueCalculator.set_rock_value(
                rock = self.ExpressionEngine.instantiate_phenotype(
                    rock = child
                )
            )

        #-----------------------------------------------------
        # SAVE THE PARENTS CLUTCH
        #-----------------------------------------------------
        parents_clutch = self.child_bred_for_parents.copy()

        #-----------------------------------------------------
        # CLEAR BREEDINGMASTERS LIST FOR NO CROSS CONTAMINATION
        #-----------------------------------------------------
        self.child_bred_for_parents.clear()

        return parents_clutch
