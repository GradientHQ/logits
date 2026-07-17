from __future__ import annotations

import os
import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    ("configured_value", "expected_value"),
    [
        (None, "0"),
        ("1", "1"),
    ],
)
def test_import_configures_tinker_telemetry_default(
    configured_value: str | None, expected_value: str
) -> None:
    env = os.environ.copy()
    if configured_value is None:
        env.pop("TINKER_TELEMETRY", None)
    else:
        env["TINKER_TELEMETRY"] = configured_value

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import os; import logits; print(os.environ['TINKER_TELEMETRY'])",
        ],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.stdout.strip() == expected_value
