# Logits Python SDK

`logits-sdk` is a thin Python facade over the official `tinker` SDK. The
distribution name on PyPI is `logits-sdk`; application code imports it as
`logits`.

It keeps the upstream runtime, type system, and request validation intact while making the Logits platform the default configuration surface:

- Prefer `LOGITS_API_KEY` and `LOGITS_BASE_URL`
- Re-export the public `tinker` API from the `logits` package
- Accept both `tinker://...` and `logits://...` checkpoint URIs
- Gracefully fall back to built-in defaults when a backend has not yet
  implemented `/api/v1/client/config`

## Installation

From git (current default — not yet on PyPI):

```bash
pip install git+https://github.com/GradientHQ/logits
```

Once published on PyPI:

```bash
pip install logits-sdk
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

- Confirm the `logits-sdk` project name is available to the Gradient PyPI account before publishing (the bare `logits` name is already taken on PyPI by an unrelated project).
- Bump both `pyproject.toml` and `src/logits/_version.py`.
- Run `pytest`, `python -m build`, and `python -m twine check dist/*`.
- Confirm GitHub Actions CI is green on the release commit.
- Configure PyPI Trusted Publishing for the `pypi` GitHub environment before publishing.
- Create a GitHub release from the version tag to trigger the publish workflow.
