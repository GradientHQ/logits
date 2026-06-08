# Logits Python SDK

`logits-sdk` is the official Python SDK for the Logits platform. Install it
under the distribution name `logits-sdk`; application code imports it as
`logits`.

## Installation

```bash
pip install logits-sdk
```

## Authentication

The SDK reads your Logits API key from the `LOGITS_API_KEY` environment variable.

## Usage

```python
import logits

service_client = logits.ServiceClient()
sampling_client = service_client.create_sampling_client(base_model="Qwen/Qwen3-8B")
future = sampling_client.sample(
    prompt=logits.ModelInput.from_ints([1, 2, 3]),
    num_samples=1,
    sampling_params=logits.SamplingParams(max_tokens=32),
)
result = future.result()
```

Async usage:

```python
import logits

service_client = logits.ServiceClient()
sampling_client = await service_client.create_sampling_client_async(
    base_model="Qwen/Qwen3-8B"
)
result = await sampling_client.sample_async(
    prompt=logits.ModelInput.from_ints([1, 2, 3]),
    num_samples=1,
    sampling_params=logits.SamplingParams(max_tokens=32),
)
```

Close the underlying holder when a long-running process no longer needs the client:

```python
service_client.holder.close()
```

## Development

Install the package and test dependencies:

```bash
python -m pip install -e .
python -m pip install pytest pytest-timeout respx
```

Run the test suite:

```bash
pytest
```

Build and validate release artifacts:

```bash
python -m pip install build twine
python -m build
python -m twine check dist/*
```

## Release Checklist

- Bump both `pyproject.toml` and `src/logits/_version.py`.
- Run `pytest`, `python -m build`, and `python -m twine check dist/*`.
- Confirm GitHub Actions CI is green on the release commit.
- Create a GitHub release from the version tag to trigger the publish workflow.
