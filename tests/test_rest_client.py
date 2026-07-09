from __future__ import annotations

import pytest

from logits._rest_client import (
    RestClient,
    _BackendCheckpoint,
    _BackendCheckpointsList,
    _to_checkpoint,
    _to_list_response,
)


@pytest.mark.parametrize(
    ("path", "expected_run", "expected_ckpt"),
    [
        # Training-state weights.
        (
            "tinker://model_6320e11a/weights/000010",
            "model_6320e11a",
            "weights/000010",
        ),
        (
            "logits://model_6320e11a/weights/000010",
            "model_6320e11a",
            "weights/000010",
        ),
        # Sampler weights (the path returned by SamplingClient.model_path).
        (
            "tinker://model_6320e11a/sampler_weights/000010",
            "model_6320e11a",
            "sampler_weights/000010",
        ),
        (
            "logits://model_6320e11a/sampler_weights/final",
            "model_6320e11a",
            "sampler_weights/final",
        ),
    ],
)
def test_parse_checkpoint_path_keeps_kind_prefix(
    path: str, expected_run: str, expected_ckpt: str
) -> None:
    run_id, checkpoint_id = RestClient._parse_checkpoint_path(path)
    assert run_id == expected_run
    assert checkpoint_id == expected_ckpt


def test_parse_checkpoint_path_rejects_non_weights_uri() -> None:
    with pytest.raises(ValueError):
        RestClient._parse_checkpoint_path("https://example.com/model")


def test_backend_checkpoint_maps_renamed_fields() -> None:
    # The Logits backend sends `logits_path` / `size` / `visibility` /
    # `expired_at`; the upstream Checkpoint type requires `tinker_path` and uses
    # `size_bytes` / `public` / `expires_at`. The raw list call would fail
    # validation without this translation.
    raw = _BackendCheckpoint.model_validate(
        {
            "checkpoint_id": "chk_abc",
            "checkpoint_type": "training",
            "name": "000010",
            "time": "2026-06-09T23:02:26.201258-07:00",
            "size": 810405661,
            "visibility": "public",
            "expired_at": None,
            "logits_path": "logits://model_6320e11a/weights/000010",
        }
    )
    ck = _to_checkpoint(raw)
    assert ck.tinker_path == "logits://model_6320e11a/weights/000010"
    assert ck.size_bytes == 810405661
    assert ck.public is True
    assert ck.expires_at is None
    assert ck.checkpoint_type == "training"


def test_backend_checkpoint_private_visibility() -> None:
    raw = _BackendCheckpoint.model_validate(
        {
            "checkpoint_id": "chk_def",
            "checkpoint_type": "sampler",
            "time": "2026-06-09T23:02:26.201258-07:00",
            "visibility": "private",
            "logits_path": "logits://model_x/sampler_weights/final",
        }
    )
    assert _to_checkpoint(raw).public is False


def test_to_list_response_builds_upstream_type() -> None:
    raw = _BackendCheckpointsList.model_validate(
        {
            "checkpoints": [
                {
                    "checkpoint_id": "chk_1",
                    "checkpoint_type": "training",
                    "time": "2026-06-09T23:02:26.201258-07:00",
                    "logits_path": "logits://run/weights/a",
                }
            ]
        }
    )
    resp = _to_list_response(raw)
    assert len(resp.checkpoints) == 1
    assert resp.checkpoints[0].checkpoint_id == "chk_1"
    assert resp.checkpoints[0].tinker_path == "logits://run/weights/a"
