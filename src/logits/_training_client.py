from __future__ import annotations

import tinker
from tinker import SamplingClient, types
from tinker.lib.client_connection_pool_type import ClientConnectionPoolType
from tinker.lib.retry_handler import RetryConfig

from ._path_compat import is_valid_model_path


class TrainingClient(tinker.TrainingClient):
    """Logits-compatible TrainingClient.

    The upstream Tinker SDK validates that `model_path` starts with `tinker://`
    when creating a SamplingClient. Some backends now return `logits://`
    checkpoints, so we accept both prefixes and pass the URI through unchanged.
    """

    def create_sampling_client(
        self, model_path: str, retry_config: RetryConfig | None = None
    ) -> SamplingClient:
        if not is_valid_model_path(model_path):
            raise ValueError("model_path must start with 'tinker://' or 'logits://'")

        async def _create_sampling_client_async() -> SamplingClient:
            # Create sampling session without the tinker:// prefix check.
            assert self.holder._sampling_client_counter is not None
            sampling_session_seq_id = self.holder._sampling_client_counter
            self.holder._sampling_client_counter += 1

            with self.holder.aclient(ClientConnectionPoolType.SESSION) as client:
                request = types.CreateSamplingSessionRequest(
                    session_id=self.holder._session_id,
                    sampling_session_seq_id=sampling_session_seq_id,
                    model_path=model_path,
                    base_model=None,
                )
                result = await client.service.create_sampling_session(request=request)

            return SamplingClient(
                self.holder,
                sampling_session_id=result.sampling_session_id,
                retry_config=retry_config,
            )

        return self.holder.run_coroutine_threadsafe(_create_sampling_client_async()).result()

    async def create_sampling_client_async(
        self, model_path: str, retry_config: RetryConfig | None = None
    ) -> SamplingClient:
        if not is_valid_model_path(model_path):
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
                    base_model=None,
                )
                result = await client.service.create_sampling_session(request=request)

            return SamplingClient(
                self.holder,
                sampling_session_id=result.sampling_session_id,
                retry_config=retry_config,
            )

        return await self.holder.run_coroutine_threadsafe(_create_sampling_client_async_inner())

