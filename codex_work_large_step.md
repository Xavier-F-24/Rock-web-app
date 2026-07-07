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

- Wire `GameMaster` into `app.py` or a small notebook flow so the playable loop uses the new split modules.
- Add relationship/inbreeding validation back into `BreedingMaster`.
- Decide final potion mechanics before balancing money and market pricing.
- Add serialization for `GameMaster`, `Inventory`, queued pairs, market pods, and pending pods.
