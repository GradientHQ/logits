from __future__ import annotations

from concurrent.futures import Future as ConcurrentFuture
from typing import Literal

from tinker import types
from tinker.lib.public_interfaces.rest_client import RestClient as TinkerRestClient

from ._path_compat import normalize_tinker_path


class RestClient(TinkerRestClient):
    """Logits-compatible RestClient.

    Upstream Tinker RestClient's `*_by_tinker_path` helpers parse the URI using
    `ParsedCheckpointTinkerPath.from_tinker_path`, which historically only
    accepted `tinker://`. Backends may now return `logits://` instead, so we
    normalize only for local parsing while still treating the rest of the API
    parameters as opaque.
    """

    def get_training_run_by_tinker_path(
        self, tinker_path: str, access_scope: Literal["owned", "accessible"] = "owned"
    ) -> ConcurrentFuture[types.TrainingRun]:
        parsed = types.ParsedCheckpointTinkerPath.from_tinker_path(normalize_tinker_path(tinker_path))
        return self.get_training_run(parsed.training_run_id, access_scope=access_scope)

    async def get_training_run_by_tinker_path_async(
        self, tinker_path: str, access_scope: Literal["owned", "accessible"] = "owned"
    ) -> types.TrainingRun:
        parsed = types.ParsedCheckpointTinkerPath.from_tinker_path(normalize_tinker_path(tinker_path))
        return await self.get_training_run_async(parsed.training_run_id, access_scope=access_scope)

    def delete_checkpoint_from_tinker_path(self, tinker_path: str) -> ConcurrentFuture[None]:
        parsed = types.ParsedCheckpointTinkerPath.from_tinker_path(normalize_tinker_path(tinker_path))
        return self._delete_checkpoint_submit(parsed.training_run_id, parsed.checkpoint_id).future()

    async def delete_checkpoint_from_tinker_path_async(self, tinker_path: str) -> None:
        parsed = types.ParsedCheckpointTinkerPath.from_tinker_path(normalize_tinker_path(tinker_path))
        await self._delete_checkpoint_submit(parsed.training_run_id, parsed.checkpoint_id)

    def get_checkpoint_archive_url_from_tinker_path(
        self, tinker_path: str
    ) -> ConcurrentFuture[types.CheckpointArchiveUrlResponse]:
        parsed = types.ParsedCheckpointTinkerPath.from_tinker_path(normalize_tinker_path(tinker_path))
        return self._get_checkpoint_archive_url_submit(parsed.training_run_id, parsed.checkpoint_id).future()

    async def get_checkpoint_archive_url_from_tinker_path_async(
        self, tinker_path: str
    ) -> types.CheckpointArchiveUrlResponse:
        parsed = types.ParsedCheckpointTinkerPath.from_tinker_path(normalize_tinker_path(tinker_path))
        return await self._get_checkpoint_archive_url_submit(parsed.training_run_id, parsed.checkpoint_id)

    def publish_checkpoint_from_tinker_path(self, tinker_path: str) -> ConcurrentFuture[None]:
        parsed = types.ParsedCheckpointTinkerPath.from_tinker_path(normalize_tinker_path(tinker_path))
        return self._publish_checkpoint_submit(parsed.training_run_id, parsed.checkpoint_id).future()

    async def publish_checkpoint_from_tinker_path_async(self, tinker_path: str) -> None:
        parsed = types.ParsedCheckpointTinkerPath.from_tinker_path(normalize_tinker_path(tinker_path))
        await self._publish_checkpoint_submit(parsed.training_run_id, parsed.checkpoint_id)

    def unpublish_checkpoint_from_tinker_path(self, tinker_path: str) -> ConcurrentFuture[None]:
        parsed = types.ParsedCheckpointTinkerPath.from_tinker_path(normalize_tinker_path(tinker_path))
        return self._unpublish_checkpoint_submit(parsed.training_run_id, parsed.checkpoint_id).future()

    async def unpublish_checkpoint_from_tinker_path_async(self, tinker_path: str) -> None:
        parsed = types.ParsedCheckpointTinkerPath.from_tinker_path(normalize_tinker_path(tinker_path))
        await self._unpublish_checkpoint_submit(parsed.training_run_id, parsed.checkpoint_id)

    def set_checkpoint_ttl_from_tinker_path(
        self, tinker_path: str, ttl_seconds: int | None
    ) -> ConcurrentFuture[None]:
        parsed = types.ParsedCheckpointTinkerPath.from_tinker_path(normalize_tinker_path(tinker_path))
        return self._set_checkpoint_ttl_submit(
            parsed.training_run_id, parsed.checkpoint_id, ttl_seconds
        ).future()

    async def set_checkpoint_ttl_from_tinker_path_async(
        self, tinker_path: str, ttl_seconds: int | None
    ) -> None:
        parsed = types.ParsedCheckpointTinkerPath.from_tinker_path(normalize_tinker_path(tinker_path))
        await self._set_checkpoint_ttl_submit(parsed.training_run_id, parsed.checkpoint_id, ttl_seconds)

