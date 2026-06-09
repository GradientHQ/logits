from __future__ import annotations

import pytest

from logits._rest_client import RestClient


@pytest.mark.parametrize(
    ("path", "expected_run", "expected_ckpt"),
    [
        # Training-state weights: the `weights/` kind prefix must be dropped so
        # the request hits the single-segment backend route.
        ("tinker://model_6320e11a/weights/000010", "model_6320e11a", "000010"),
        ("logits://model_6320e11a/weights/000010", "model_6320e11a", "000010"),
        # Sampler weights (the path returned by SamplingClient.model_path).
        ("tinker://model_6320e11a/sampler_weights/000010", "model_6320e11a", "000010"),
        ("logits://model_6320e11a/sampler_weights/final", "model_6320e11a", "final"),
    ],
)
def test_parse_checkpoint_path_strips_kind_prefix(
    path: str, expected_run: str, expected_ckpt: str
) -> None:
    run_id, checkpoint_id = RestClient._parse_checkpoint_path(path)
    assert run_id == expected_run
    assert checkpoint_id == expected_ckpt
    # The backend route `/checkpoints/{checkpoint_id}/...` matches a single path
    # segment; a slash here is what produced the plain-text 404.
    assert "/" not in checkpoint_id


def test_parse_checkpoint_path_rejects_non_weights_uri() -> None:
    with pytest.raises(ValueError):
        RestClient._parse_checkpoint_path("https://example.com/model")
