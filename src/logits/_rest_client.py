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

    @staticmethod
    def _parse_checkpoint_path(tinker_path: str) -> tuple[str, str]:
        """Return ``(training_run_id, checkpoint_id)`` for a weights URI.

        The Logits backend addresses a checkpoint by its bare name -- a single
        path segment such as ``000010`` -- under
        ``/api/v1/training_runs/{run}/checkpoints/{checkpoint_id}/...``.

        ``ParsedCheckpointTinkerPath`` follows the upstream Tinker convention of
        keeping the checkpoint kind inside ``checkpoint_id``, so it yields
        ``weights/000010`` or ``sampler_weights/000010``. Sending that slash
        unencoded expands the URL to ``.../checkpoints/weights/000010/...``,
        which no single-segment route matches, so the gateway answers with a
        plain-text ``404 page not found``. Strip the kind prefix so the request
        targets the real route.
        """
        parsed = types.ParsedCheckpointTinkerPath.from_tinker_path(
            normalize_tinker_path(tinker_path)
        )
        checkpoint_id = parsed.checkpoint_id.rsplit("/", 1)[-1]
        return parsed.training_run_id, checkpoint_id

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
