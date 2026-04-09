# Logits Python SDK

`logits` is a thin Python facade over the official `tinker` SDK.

It keeps the upstream runtime, type system, and request validation intact while making the Logits platform the default configuration surface:

- Prefer `LOGITS_API_KEY` and `LOGITS_BASE_URL`
- Re-export the public `tinker` API from the `logits` package
- Keep `tinker://...` checkpoint paths and all upstream client behavior

Typical usage:

```python
import logits

service_client = logits.ServiceClient()
sampling_client = service_client.create_sampling_client(base_model="Qwen/Qwen3-8B")
result = sampling_client.sample(
    prompt=logits.ModelInput.from_ints([1, 2, 3]),
    num_samples=1,
    sampling_params=logits.SamplingParams(max_tokens=32),
)
```

The `tinker` package remains an implementation dependency, but application code should import `logits`.
