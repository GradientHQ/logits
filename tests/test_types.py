from __future__ import annotations

from logits.types import LossFnType, TensorData
from logits.types.tensor_data import TensorData as TensorDataModule
from tinker.types.tensor_data import TensorData as TinkerTensorData


def test_types_package_reexports_tinker_types() -> None:
    assert TensorData is TinkerTensorData
    assert TensorDataModule is TinkerTensorData
    assert LossFnType.__module__ == "logits.types"
