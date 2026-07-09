from __future__ import annotations

from concurrent.futures import Future as ConcurrentFuture
from datetime import datetime
from typing import Literal

from tinker import types
from tinker._models import BaseModel as _SdkBaseModel
from tinker.lib.client_connection_pool_type import ClientConnectionPoolType
from tinker.lib.public_interfaces.api_future import AwaitableConcurrentFuture
from tinker.lib.public_interfaces.rest_client import RestClient as TinkerRestClient

from ._path_compat import normalize_tinker_path


class _BackendCheckpoint(_SdkBaseModel):
    """Mirror of the checkpoint object the Logits backend actually returns.

    The upstream Tinker ``Checkpoint`` model requires ``tinker_path`` and uses
    ``size_bytes`` / ``public`` / ``expires_at``. The Logits backend instead
    sends ``logits_path``, ``size``, ``visibility`` and ``expired_at`` (and a
    human-readable ``name``). Because the upstream model marks ``tinker_path``
    as required, the raw list call fails validation before we ever see the
    data. We parse the backend's real field names here and translate back to the
    upstream type so callers keep the documented ``Checkpoint`` shape.
    """

    checkpoint_id: str
    checkpoint_type: str | None = None
    name: str | None = None
    time: datetime | None = None
    logits_path: str | None = None
    tinker_path: str | None = None
    size: int | None = None
    size_bytes: int | None = None
    visibility: str | None = None
    public: bool | None = None
    expired_at: datetime | None = None
    expires_at: datetime | None = None


class _BackendCheckpointsList(_SdkBaseModel):
    checkpoints: list[_BackendCheckpoint] = []
    cursor: object | None = None


def _to_checkpoint(b: _BackendCheckpoint) -> types.Checkpoint:
    public = b.public if b.public is not None else (b.visibility == "public")
    return types.Checkpoint(
        checkpoint_id=b.checkpoint_id,
        checkpoint_type=b.checkpoint_type or "training",  # type: ignore[arg-type]
        time=b.time,  # type: ignore[arg-type]
        tinker_path=(b.tinker_path or b.logits_path or ""),
        size_bytes=(b.size_bytes if b.size_bytes is not None else b.size),
        public=bool(public),
        expires_at=(b.expires_at or b.expired_at),
    )


def _to_list_response(raw: _BackendCheckpointsList) -> types.CheckpointsListResponse:
    cursor = None
    if isinstance(raw.cursor, dict):
        try:
            cursor = types.Cursor(**raw.cursor)
        except Exception:
            cursor = None
    return types.CheckpointsListResponse(
        checkpoints=[_to_checkpoint(b) for b in raw.checkpoints],
        cursor=cursor,
    )


class RestClient(TinkerRestClient):
    """Logits-compatible RestClient.

    Upstream Tinker RestClient's `*_by_tinker_path` helpers parse the URI using
    `ParsedCheckpointTinkerPath.from_tinker_path`, which historically only
    accepted `tinker://`. Backends may now return `logits://` instead, so we
    normalize only for local parsing while still treating the rest of the API
    parameters as opaque.
    """

    @staticmethod
    def _parse_checkpoint_path(tinker_path: str) -> tuple[str, str]:
        """Return ``(training_run_id, checkpoint_id)`` for a weights URI.

        ``ParsedCheckpointTinkerPath`` follows the upstream Tinker convention of
        keeping the checkpoint kind inside ``checkpoint_id``, so it yields
        ``weights/000010`` or ``sampler_weights/000010``. The Logits backend
        accepts these typed checkpoint references directly.
        """
        parsed = types.ParsedCheckpointTinkerPath.from_tinker_path(
            normalize_tinker_path(tinker_path)
        )
        return parsed.training_run_id, parsed.checkpoint_id

    # ------------------------------------------------------------------
    # Checkpoint listing (backend returns `logits_path`, not `tinker_path`)
    # ------------------------------------------------------------------
    async def _fetch_run_checkpoints(self, training_run_id: types.ModelID) -> _BackendCheckpointsList:
        async def _send() -> _BackendCheckpointsList:
            with self.holder.aclient(ClientConnectionPoolType.TRAIN) as client:
                return await client.get(
                    f"/api/v1/training_runs/{training_run_id}/checkpoints",
                    cast_to=_BackendCheckpointsList,
                )

        return await self.holder.execute_with_retries(_send)

    async def _fetch_user_checkpoints(self, limit: int, offset: int) -> _BackendCheckpointsList:
        async def _send() -> _BackendCheckpointsList:
            with self.holder.aclient(ClientConnectionPoolType.TRAIN) as client:
                return await client.get(
                    "/api/v1/checkpoints",
                    options={"params": {"limit": limit, "offset": offset}},
                    cast_to=_BackendCheckpointsList,
                )

        return await self.holder.execute_with_retries(_send)

    def _list_checkpoints_submit(
        self, training_run_id: types.ModelID
    ) -> AwaitableConcurrentFuture[types.CheckpointsListResponse]:
        async def _coro() -> types.CheckpointsListResponse:
            return _to_list_response(await self._fetch_run_checkpoints(training_run_id))

        return self.holder.run_coroutine_threadsafe(_coro())

    def _list_user_checkpoints_submit(
        self, limit: int = 100, offset: int = 0
    ) -> AwaitableConcurrentFuture[types.CheckpointsListResponse]:
        async def _coro() -> types.CheckpointsListResponse:
            return _to_list_response(await self._fetch_user_checkpoints(limit, offset))

        return self.holder.run_coroutine_threadsafe(_coro())

    def publish_checkpoint_from_tinker_path(self, tinker_path: str) -> ConcurrentFuture[None]:
        training_run_id, checkpoint_id = self._parse_checkpoint_path(tinker_path)
        return self._publish_checkpoint_submit(training_run_id, checkpoint_id).future()

    async def publish_checkpoint_from_tinker_path_async(self, tinker_path: str) -> None:
        training_run_id, checkpoint_id = self._parse_checkpoint_path(tinker_path)
        await self._publish_checkpoint_submit(training_run_id, checkpoint_id)

    def unpublish_checkpoint_from_tinker_path(self, tinker_path: str) -> ConcurrentFuture[None]:
        training_run_id, checkpoint_id = self._parse_checkpoint_path(tinker_path)
        return self._unpublish_checkpoint_submit(training_run_id, checkpoint_id).future()

    async def unpublish_checkpoint_from_tinker_path_async(self, tinker_path: str) -> None:
        training_run_id, checkpoint_id = self._parse_checkpoint_path(tinker_path)
        await self._unpublish_checkpoint_submit(training_run_id, checkpoint_id)

    # ------------------------------------------------------------------
    # Path helpers unchanged from the original Logits shim
    # ------------------------------------------------------------------
    def get_training_run_by_tinker_path(
        self, tinker_path: str, access_scope: Literal["owned", "accessible"] = "owned"
    ) -> ConcurrentFuture[types.TrainingRun]:
        training_run_id, _ = self._parse_checkpoint_path(tinker_path)
        return self.get_training_run(training_run_id, access_scope=access_scope)

    async def get_training_run_by_tinker_path_async(
        self, tinker_path: str, access_scope: Literal["owned", "accessible"] = "owned"
    ) -> types.TrainingRun:
        training_run_id, _ = self._parse_checkpoint_path(tinker_path)
        return await self.get_training_run_async(training_run_id, access_scope=access_scope)

    def delete_checkpoint_from_tinker_path(self, tinker_path: str) -> ConcurrentFuture[None]:
        training_run_id, checkpoint_id = self._parse_checkpoint_path(tinker_path)
        return self._delete_checkpoint_submit(training_run_id, checkpoint_id).future()

    async def delete_checkpoint_from_tinker_path_async(self, tinker_path: str) -> None:
        training_run_id, checkpoint_id = self._parse_checkpoint_path(tinker_path)
        await self._delete_checkpoint_submit(training_run_id, checkpoint_id)

    def get_checkpoint_archive_url_from_tinker_path(
        self, tinker_path: str
    ) -> ConcurrentFuture[types.CheckpointArchiveUrlResponse]:
        training_run_id, checkpoint_id = self._parse_checkpoint_path(tinker_path)
        return self._get_checkpoint_archive_url_submit(training_run_id, checkpoint_id).future()

    async def get_checkpoint_archive_url_from_tinker_path_async(
        self, tinker_path: str
    ) -> types.CheckpointArchiveUrlResponse:
        training_run_id, checkpoint_id = self._parse_checkpoint_path(tinker_path)
        return await self._get_checkpoint_archive_url_submit(training_run_id, checkpoint_id)

    def set_checkpoint_ttl_from_tinker_path(
        self, tinker_path: str, ttl_seconds: int | None
    ) -> ConcurrentFuture[None]:
        training_run_id, checkpoint_id = self._parse_checkpoint_path(tinker_path)
        return self._set_checkpoint_ttl_submit(
            training_run_id, checkpoint_id, ttl_seconds
        ).future()

    async def set_checkpoint_ttl_from_tinker_path_async(
        self, tinker_path: str, ttl_seconds: int | None
    ) -> None:
        training_run_id, checkpoint_id = self._parse_checkpoint_path(tinker_path)
        await self._set_checkpoint_ttl_submit(training_run_id, checkpoint_id, ttl_seconds)
