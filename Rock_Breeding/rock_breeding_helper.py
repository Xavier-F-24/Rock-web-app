
import random
from typing import List

name_bits_start = [
    "Grum", "Peb", "Bas", "Quartz", "Moss", "Igni", "Crag", "Glim", "Obsi", "Feld"
]

name_bits_end = [
    "ble", "ite", "or", "yx", "stone", "ling", "rock", "spar", "gem", "oid"
]

def random_rock_name() -> str:
    return random.choice(name_bits_start) + random.choice(name_bits_end)

