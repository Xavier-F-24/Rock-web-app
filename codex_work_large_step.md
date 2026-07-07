# Codex Work Large Step

## What Changed

- Committed the pytest work as `e89502b7` with message `test implementation`.
- Added `GenomeFactory.make_selected_rock_genome()` so prototype systems can create rocks with requested allele pairs.
- Reworked `Rock_Market/rock_market_helper.py` around a `MarketManager` dataclass.
- Reworked `Rock_GameState/rock_game_state_helper.py` around a `GameMaster` dataclass.
- Added `Develepor_X/test_game_prototype.py` to smoke-test starting a game, breeding, generation advance, selling, potions, requested rocks, and market pods.

## Prototype Flow Now Supported

- Instantiate a playable game with `GameMaster()` or `create_new_game(seed=...)`.
- Starter rocks are created automatically by default; call `GameMaster(auto_start=False)` if you want an empty shell.
- Queue breeding with `game.add_pair_to_queue(parent_a_id, parent_b_id, potion_key=None)`.
- Breed and advance with `game.advance_generation()`.
- Inspect display state with `game.update_display()` and rock lines with `game.show_rocks()`.
- Enter the market through `game.market_manager` or wrappers:
  - `game.buy_potion("fertility")`
  - `game.sell_rock(rock_id)`
  - `game.buy_random_rock()`
  - `game.buy_defined_trait_rock({"color": "34", "eyes": "11"})`
  - `game.market_manager.buy_market_pod(game, offer_id)`
  - `game.market_manager.choose_market_pod_child(game, 0)`

## Questions / Possible Problems

- I replaced the old Market and GameState helper bodies because they were not import-safe in the split-module world. If the Streamlit app still imports from `rockgame_core.py`, the app may not use these new managers yet.
- `GameMaster.advance_generation()` both breeds the queue and advances the generation. If you want a separate "breed but do not advance" phase, we can split that into a stricter turn pipeline.
- Potion effects are intentionally simple in this pass. Mutation and anti-craisen change probabilities, fertility adds one child, and reroll currently makes the pair safer rather than fully rerolling clutch math.
- Market pods currently generate one offer per tier each refresh. The old market file had random tier counts, but a predictable first prototype is easier to test and tune.
- Guest market parents are inserted into the game tree as non-owned rocks when a pod is bought. They are currently marked through `is_market=True`; we may want an explicit ownership/status enum later.
- Requested/defined-trait rocks accept direct allele pairs, not friendly labels yet. Example: `"color": "34"` creates orange; UI-friendly catalog selection can be layered on top.

## Next Suggested Work

- Wire the split-module prototype mode deeper into the main Streamlit UI once the old `rockgame_core.py` flow is retired.
- Add serialization tests for older save compatibility if we want to migrate legacy saves.
- Tune potion prices and market prices after a few full playthroughs.

## Continuing Work: Split Prototype Integration

- Added a Streamlit sidebar toggle, `Use split-module prototype`, that launches a compact `GameMaster`-backed playable loop in `app.py`.
- Added relationship/inbreeding validation to `BreedingMaster` for self, same sex, inactive rocks, parent/child, siblings, ancestor/descendant, and shared ancestor cases when a game context is supplied.
- Locked first-pass potion mechanics:
  - `fertility`: adds one child to clutch size.
  - `reroll`: rolls clutch twice and keeps the larger clutch.
  - `mutation`: raises child mutation chance from `0.01` to `0.12`.
  - `anti_craisen`: reduces craisen chance to `0.0` for the pair.
- Rebuilt split-module serialization in `Rock_Serialization/rock_serialization_helper.py` for `GameMaster`, `Inventory`, `QueuedPair`, `MarketPodOffer`, `PendingMarketPod`, rocks, genomes, and rock names.
- Added tests for serialization, relationship validation, and final first-pass potion settings.

## New Questions / Possible Problems

- The split-module Streamlit mode is intentionally compact and table-first. It proves the playable loop, but it does not yet use the richer rock cards/tree UI from the legacy app.
- Relationship validation currently blocks any shared ancestor. That is conservative; if we later want cousins or distant relatives allowed with penalties, this should become a relatedness score instead of a hard block.
- Pending market pods serialize full parent/child rock data. This is simple and robust, but it duplicates parent data that may also exist in `rock_list`.
- Legacy save files from `rockgame_core.py` are not supported by the new serializer yet.
