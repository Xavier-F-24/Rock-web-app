# Rock Genetics Game

A Streamlit app for breeding, importing, selling, saving, loading, and drawing lineage trees for the split-module rock genetics game.

## Persistent Rock World

Every new playable game now creates a persistent shared world containing the
player farm and either 2–12 selected NPC farms or a seeded random 3–8. NPC
farms begin with genuine simulated lineages one to three generations ahead.
Every world rock has one owner and globally unique ID.

Use **Rock World** to inspect public farmer galleries and lineage trees or send
a direct trade offer. Use **Market** for farmer listings, real sibling family
pods, player listings, outgoing offers, the farmer inbox, and potions. Synthetic
imports and instant rock sales are disabled in persistent games. Press **End
World Turn** when the NPC farmers should each take one action.

Save version `0.4.0` includes the complete world economy. Older saves receive a
deterministically generated farmer world without changing existing player
rocks, money, generation, pending legacy market state, or game rules.

## Run Locally

Install runtime dependencies, then start the Streamlit app:

```powershell
python -m pip install -r requirements.txt
streamlit run streamlit_app.py
```

The app entrypoint is `streamlit_app.py`.

## Streamlit Community Cloud

Use these deployment settings:

- Repository: this repo
- Branch: your deployment branch
- Main file path: `streamlit_app.py`
- Python version: 3.12 recommended

No `packages.txt` is required right now. The app uses Python packages only.

## Dependencies

Runtime dependencies are listed in `requirements.txt`:

- `streamlit`
- `numpy`
- `matplotlib`
- `plotly`

Test-only dependencies live in `Develepor_X/requirements-dev.txt` and are not required for Streamlit Cloud deployment.

## Troubleshooting

If Streamlit Cloud reports a missing module, confirm it is a third-party package and add it to `requirements.txt`. Do not add built-in Python libraries such as `json`, `random`, `dataclasses`, `pathlib`, or `typing`.

If the app starts locally but not on Streamlit Cloud, check that the Cloud entrypoint is exactly `streamlit_app.py` and that files are imported using repo-relative Python package paths.

If the tree or rock images look stale after loading a save, refresh the page. Saves do not store rendered PNG data; rock images are regenerated from game state at runtime.
