# Rock AI

The gameplay agents use a strict player-visible observation contract. Hidden genotypes and death genes are available only to oracle label generation, evaluation, and the Observatory's explicitly labeled developer display.

## Player-Like Training

Generate player-visible pair-ranking data:

```powershell
python -m Rock_AI.scripts.generate_pair_ranking_dataset --farms 20 --trials-per-pair 25 --seed 1234 --observation player --output training_data/player_pair_ranker_smoke
```

Train the supervised PyTorch ranker:

```powershell
python -m Rock_AI.scripts.train_pair_ranker --dataset training_data/player_pair_ranker_smoke --output training_runs/player_pair_ranker_smoke --epochs 10 --seed 1234
```

Evolve a separate feed-forward NEAT ranker:

```powershell
python -m Rock_AI.scripts.train_neat_pair_ranker --dataset training_data/player_pair_ranker_smoke --output training_runs/neat_pair_ranker_smoke --population 100 --generations 10 --seed 1234
```

NEAT training uses rotating fitness scenarios, fixed validation scenarios, and one fixed replay-only showcase. Champion networks are exported as safe JSON; Streamlit never loads NEAT pickle checkpoints.

## Observatory

Run `streamlit run streamlit_app.py`, then open **AI Observatory**. The page provides:

- **Agent Session** for live stepping and campaign control.
- **Network** for observable activations and local edge signals.
- **Training Replay** for fixed-seed champion comparison across NEAT generations.

The network display is diagnostic telemetry, not hidden model reasoning. Local edge signals are calculated as source activation multiplied by connection weight.

## Tests

```powershell
python -m pytest Rock_AI/tests tests
```
