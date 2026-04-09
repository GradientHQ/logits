from __future__ import annotations

from typing import Any

import tinker
from tinker import SamplingClient, types
from tinker.lib.client_connection_pool_type import ClientConnectionPoolType
from tinker.lib.retry_handler import RetryConfig

from ._config import resolve_service_client_kwargs
from ._path_compat import is_valid_model_path
from ._training_client import TrainingClient as LogitsTrainingClient
from ._rest_client import RestClient as LogitsRestClient


class ServiceClient(tinker.ServiceClient):
    """Logits-first facade over ``tinker.ServiceClient``."""

    def __init__(
        self,
        user_metadata: dict[str, str] | None = None,
        project_id: str | None = None,
        **kwargs: Any,
    ) -> None:
        base_url = kwargs.pop("base_url", None)
        api_key = kwargs.pop("api_key", None)
        default_headers = kwargs.pop("default_headers", None)
        translated_kwargs = resolve_service_client_kwargs(
            api_key=api_key,
            base_url=base_url,
            default_headers=default_headers,
        )
        super().__init__(
            user_metadata=user_metadata,
            project_id=project_id,
            **translated_kwargs,
            **kwargs,
        )

    @staticmethod
    def _wrap_training_client(training_client: tinker.TrainingClient) -> LogitsTrainingClient:
        # tinker.TrainingClient stores the model seq id in a private field.
        # We only need it to construct an equivalent wrapper instance.
        return LogitsTrainingClient(
            training_client.holder,
            getattr(training_client, "_training_client_id"),
            training_client.model_id,
        )

    def create_sampling_client(
        self,
        model_path: str | None = None,
        base_model: str | None = None,
        retry_config: RetryConfig | None = None,
    ) -> SamplingClient:
        if model_path is None and base_model is None:
            raise ValueError("Either model_path or base_model must be provided")
        if model_path is not None and not is_valid_model_path(model_path):
            raise ValueError("model_path must start with 'tinker://' or 'logits://'")

        async def _create_sampling_client_async() -> SamplingClient:
            assert self.holder._sampling_client_counter is not None
            sampling_session_seq_id = self.holder._sampling_client_counter
            self.holder._sampling_client_counter += 1

            with self.holder.aclient(ClientConnectionPoolType.SESSION) as client:
                request = types.CreateSamplingSessionRequest(
                    session_id=self.holder._session_id,
                    sampling_session_seq_id=sampling_session_seq_id,
                    model_path=model_path,
                    base_model=base_model,
                )
                result = await client.service.create_sampling_session(request=request)

            return SamplingClient(
                self.holder,
                sampling_session_id=result.sampling_session_id,
                retry_config=retry_config,
            )

        return self.holder.run_coroutine_threadsafe(_create_sampling_client_async()).result()

    async def create_sampling_client_async(
        self,
        model_path: str | None = None,
        base_model: str | None = None,
        retry_config: RetryConfig | None = None,
    ) -> SamplingClient:
        if model_path is None and base_model is None:
            raise ValueError("Either model_path or base_model must be provided")
        if model_path is not None and not is_valid_model_path(model_path):
            raise ValueError("model_path must start with 'tinker://' or 'logits://'")

        async def _create_sampling_client_async_inner() -> SamplingClient:
            assert self.holder._sampling_client_counter is not None
            sampling_session_seq_id = self.holder._sampling_client_counter
            self.holder._sampling_client_counter += 1

            with self.holder.aclient(ClientConnectionPoolType.SESSION) as client:
                request = types.CreateSamplingSessionRequest(
                    session_id=self.holder._session_id,
                    sampling_session_seq_id=sampling_session_seq_id,
                    model_path=model_path,
                    base_model=base_model,
                )
                result = await client.service.create_sampling_session(request=request)

            return SamplingClient(
                self.holder,
                sampling_session_id=result.sampling_session_id,
                retry_config=retry_config,
            )

        return await self.holder.run_coroutine_threadsafe(_create_sampling_client_async_inner())

    def create_lora_training_client(
        self,
        base_model: str,
        rank: int = 32,
        seed: int | None = None,
        train_mlp: bool = True,
        train_attn: bool = True,
        train_unembed: bool = True,
        user_metadata: dict[str, str] | None = None,
    ) -> LogitsTrainingClient:
        training_client = super().create_lora_training_client(
            base_model=base_model,
            rank=rank,
            seed=seed,
            train_mlp=train_mlp,
            train_attn=train_attn,
            train_unembed=train_unembed,
            user_metadata=user_metadata,
        )
        return self._wrap_training_client(training_client)

    async def create_lora_training_client_async(
        self,
        base_model: str,
        rank: int = 32,
        seed: int | None = None,
        train_mlp: bool = True,
        train_attn: bool = True,
        train_unembed: bool = True,
        user_metadata: dict[str, str] | None = None,
    ) -> LogitsTrainingClient:
        training_client = await super().create_lora_training_client_async(
            base_model=base_model,
            rank=rank,
            seed=seed,
            train_mlp=train_mlp,
            train_attn=train_attn,
            train_unembed=train_unembed,
            user_metadata=user_metadata,
        )
        return self._wrap_training_client(training_client)

    def create_training_client_from_state(
        self, path: str, user_metadata: dict[str, str] | None = None
    ) -> LogitsTrainingClient:
        training_client = super().create_training_client_from_state(
            path, user_metadata=user_metadata
        )
        return self._wrap_training_client(training_client)

    async def create_training_client_from_state_async(
        self, path: str, user_metadata: dict[str, str] | None = None
    ) -> LogitsTrainingClient:
        training_client = await super().create_training_client_from_state_async(
            path, user_metadata=user_metadata
        )
        return self._wrap_training_client(training_client)

    def create_training_client_from_state_with_optimizer(
        self, path: str, user_metadata: dict[str, str] | None = None
    ) -> LogitsTrainingClient:
        training_client = super().create_training_client_from_state_with_optimizer(
            path, user_metadata=user_metadata
        )
        return self._wrap_training_client(training_client)

    async def create_training_client_from_state_with_optimizer_async(
        self, path: str, user_metadata: dict[str, str] | None = None
    ) -> LogitsTrainingClient:
        training_client = await super().create_training_client_from_state_with_optimizer_async(
            path, user_metadata=user_metadata
        )
        return self._wrap_training_client(training_client)

    def create_rest_client(self) -> LogitsRestClient:
        return LogitsRestClient(self.holder)


def create_service_client(**kwargs: Any) -> ServiceClient:
    return ServiceClient(**kwargs)
