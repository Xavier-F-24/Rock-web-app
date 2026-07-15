from __future__ import annotations

from pathlib import Path

from Rock_Streamlit.components.checkpoint_selector_helper import (
    discover_pair_ranker_checkpoints,
)


def test_checkpoint_discovery_pairs_predictor_backed_rankers(tmp_path):
    standalone = tmp_path / "training_runs" / "pair_ranker_standalone" / "best.pt"
    backed = tmp_path / "training_runs" / "pair_ranker_backed" / "best.pt"
    predictor = tmp_path / "training_runs" / "breeding_predictor_demo" / "best.pt"
    for path in (standalone, backed, predictor):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()

    def metadata(path: Path):
        if "breeding_predictor" in path.as_posix():
            return {"target_dimension": 12}
        if "backed" in path.as_posix():
            return {"predictor_feature_dimension": 12, "epoch": 3}
        return {"predictor_feature_dimension": 0, "epoch": 2}

    options = discover_pair_ranker_checkpoints(tmp_path, metadata_loader=metadata)
    assert len(options) == 2
    assert options[0].predictor_path is None
    assert options[1].predictor_path.endswith("breeding_predictor_demo/best.pt")


def test_incomplete_predictor_backed_ranker_is_hidden(tmp_path):
    ranker = tmp_path / "training_runs" / "pair_ranker_backed" / "best.pt"
    ranker.parent.mkdir(parents=True)
    ranker.touch()

    options = discover_pair_ranker_checkpoints(
        tmp_path,
        metadata_loader=lambda path: {"predictor_feature_dimension": 5},
    )
    assert options == ()
