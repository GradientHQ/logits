# Logits Python SDK

`logits` is a thin Python facade over the official `tinker` SDK.

It keeps the upstream runtime, type system, and request validation intact while making the Logits platform the default configuration surface:

- Prefer `LOGITS_API_KEY` and `LOGITS_BASE_URL`
- Re-export the public `tinker` API from the `logits` package
- Keep `tinker://...` checkpoint paths and all upstream client behavior

## Installation

```bash
pip install logits
```

The `tinker` package is installed as the implementation dependency, but application code should import `logits`.

## Authentication

Set a Logits API key before creating clients:

```bash
export LOGITS_API_KEY="your-api-key"
```

For non-default deployments, set a base URL:

```bash
export LOGITS_BASE_URL="https://api.example.com"
```

`TINKER_API_KEY` and `TINKER_BASE_URL` remain supported as fallback environment variables for compatibility.

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

Async clients follow the upstream `tinker` async API:

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

- Confirm the `logits` project name is available to the Gradient PyPI account before publishing.
- Bump both `pyproject.toml` and `src/logits/_version.py`.
- Run `pytest`, `python -m build`, and `python -m twine check dist/*`.
- Confirm GitHub Actions CI is green on the release commit.
- Configure PyPI Trusted Publishing for the `pypi` GitHub environment before publishing.
- Create a GitHub release from the version tag to trigger the publish workflow.
