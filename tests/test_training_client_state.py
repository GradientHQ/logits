from __future__ import annotations

import asyncio
import json

import httpx
import pytest
import respx
import tinker

import logits


@pytest.fixture
def wire_requests():
    requests = []
    results = {}

    def handle(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else {}
        endpoint = request.url.path
        requests.append((endpoint, body))
        if endpoint == "/api/v1/client/config":
            result = {}
        elif endpoint == "/api/v1/create_session":
            result = {"session_id": "local-session"}
        elif endpoint == "/api/v1/weights_info":
            result = {"base_model": "Qwen/Qwen3-8B", "is_lora": True, "lora_rank": 32}
        elif endpoint == "/api/v1/create_model":
            results["create"] = {"model_id": "local-model", "type": "create_model"}
            result = {"request_id": "create"}
        elif endpoint == "/api/v1/load_weights":
            results["load"] = {"path": body["path"], "type": "load_weights"}
            result = {"request_id": "load"}
        elif endpoint == "/api/v1/optim_step":
            results["optim"] = {"metrics": {}}
            result = {"request_id": "optim"}
        elif endpoint == "/api/v1/retrieve_future":
            result = results[body["request_id"]]
        else:
            raise AssertionError(f"Unexpected request: {endpoint}")
        return httpx.Response(200, json=result)

    with respx.mock(assert_all_called=False) as router:
        router.route(host="logits.test").mock(side_effect=handle)
        yield requests


@pytest.mark.parametrize("sdk", [tinker, logits], ids=["tinker-control", "logits"])
def test_fresh_training_client_starts_at_sequence_one(sdk, wire_requests):
    service = sdk.ServiceClient(base_url="http://logits.test", api_key="tml-fake-key")
    try:
        training = service.create_lora_training_client(base_model="Qwen/Qwen3-8B")
        training.optim_step(sdk.AdamParams(learning_rate=1e-4)).result()
        assert [
            body["seq_id"]
            for endpoint, body in wire_requests
            if endpoint == "/api/v1/optim_step"
        ] == [1]
    finally:
        service.holder.close()


@pytest.mark.parametrize("sdk", [tinker, logits], ids=["tinker-control", "logits"])
@pytest.mark.parametrize(
    "optimizer", [False, True], ids=["weights", "weights-and-optimizer"]
)
@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("scheme", ["tinker", "logits"])
def test_restored_training_client_continues_sequence(
    sdk, optimizer, asynchronous, scheme, wire_requests
):
    service = sdk.ServiceClient(base_url="http://logits.test", api_key="tml-fake-key")
    method = "create_training_client_from_state" + (
        "_with_optimizer" if optimizer else ""
    )
    checkpoint = f"{scheme}://previous-run/weights/checkpoint"
    try:
        if asynchronous:

            async def run():
                training = await getattr(service, method + "_async")(checkpoint)
                future = await training.optim_step_async(
                    sdk.AdamParams(learning_rate=1e-4)
                )
                await future.result_async()

            asyncio.run(run())
        else:
            training = getattr(service, method)(checkpoint)
            training.optim_step(sdk.AdamParams(learning_rate=1e-4)).result()
        sequences = [
            (endpoint, body["seq_id"])
            for endpoint, body in wire_requests
            if endpoint in ("/api/v1/load_weights", "/api/v1/optim_step")
        ]
        assert sequences == [("/api/v1/load_weights", 1), ("/api/v1/optim_step", 2)]
    finally:
        service.holder.close()
