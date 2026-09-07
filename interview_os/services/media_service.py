"""Audio transcription, recording, speech feedback, and synthesis operations."""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import logging
import os
import re
import stat
import wave
from datetime import datetime, timedelta, timezone
from functools import partial
from pathlib import Path
from time import perf_counter
from typing import Any, BinaryIO
from uuid import UUID, uuid4

from interview_os.core.debug import DebugLevel
from interview_os.core.evidence import EvidenceSource
from interview_os.core.request_context import current_owner
from interview_os.core.runtime import AgentRuntime
from interview_os.core.state import (
    LIVE_AUDIO_CANCELLED_CAPTURE_MAX,
    LIVE_AUDIO_CAPTURE_DRAIN_GRACE_SECONDS,
    LIVE_AUDIO_CAPTURE_LEASE_TTL_SECONDS,
    LIVE_AUDIO_PENDING_CAPTURE_MAX,
    LIVE_AUDIO_RECEIPT_MAX,
    LIVE_RECENT_SEGMENT_WINDOW,
    OMNI_CONTEXT_CHAR_LIMIT,
    InterviewState,
    LiveAudioCaptureReceipt,
    LiveAudioPart,
    LiveInterviewStatus,
    MockAnswerDraft,
    QuestionSuggestion,
    QuestionSuggestionType,
    SpeechDeliveryFeedback,
    TranscriptSpeaker,
)
from interview_os.tools.asr import ASRError
from interview_os.tools.tts import TTSError

logger = logging.getLogger(__name__)

from interview_os.services.service_contracts import (
    SPEECH_FEEDBACK_PROHIBITED_TERMS,
    LiveInterviewStateError,
    MockInterviewStateError,
    WorkflowExecutionError,
)
from interview_os.services.service_mixin import InterviewServiceMixin


class MediaServiceMixin(InterviewServiceMixin):
    """Audio transcription, recording, speech feedback, and synthesis operations."""

    @staticmethod
    def _file_sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _write_private_bytes(path: Path, content: bytes) -> None:
        """Create/truncate one non-symlink recording with owner-only access."""

        flags = os.O_CREAT | os.O_TRUNC | os.O_WRONLY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(path, flags, 0o600)
        try:
            if hasattr(os, "fchmod"):
                os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "wb") as stream:
                descriptor = -1
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
        finally:
            if descriptor >= 0:
                os.close(descriptor)

    @staticmethod
    def _write_then_replace(temp: Path, target: Path, content: bytes) -> None:
        """Write bytes atomically and never retain a failed plaintext temp file."""

        try:
            MediaServiceMixin._write_private_bytes(temp, content)
            os.replace(temp, target)
        except OSError:
            try:
                temp.unlink(missing_ok=True)
            except OSError:
                logger.warning("Could not remove failed audio temp file %s", temp.name)
            raise

    def _ensure_private_recordings_dir(self) -> None:
        """Create and repair the dedicated audio directory's local permissions."""

        self._recordings_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        metadata = self._recordings_dir.lstat()
        if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
            raise OSError("recordings path must be a dedicated local directory")
        os.chmod(self._recordings_dir, 0o700)
        for candidate in self._recordings_dir.iterdir():
            try:
                mode = candidate.lstat().st_mode
                if stat.S_ISREG(mode):
                    os.chmod(candidate, 0o600, follow_symlinks=False)
            except (FileNotFoundError, OSError, NotImplementedError):
                logger.warning(
                    "Could not restrict existing recording permissions for %s",
                    candidate.name,
                )

    def set_live_audio_mode(self, mode: str) -> None:
        if mode not in {"asr_text", "audio_direct"}:
            raise ValueError(f"Unsupported live audio mode: {mode}")
        if mode == "audio_direct" and self.omni_client is None:
            raise ValueError("Live audio direct requires an omni client")
        self.live_audio_mode = mode

    def refresh_audio_settings_etag(self) -> str:
        """Rotate the opaque generation token for effective audio settings.

        A random epoch avoids restart ABA without exposing a hash that could be
        used as an offline oracle for a low-entropy API key.
        """

        self.audio_settings_etag = uuid4().hex
        return self.audio_settings_etag

    async def _acquire_audio_settings_lease(self) -> str:
        """Fence one provider operation from an in-place settings mutation."""

        async with self._settings_lock:
            generation = self.audio_settings_etag
            self._active_audio_settings_leases += 1
            return generation

    def _release_audio_settings_lease(self) -> None:
        """Release an event-loop-local lease without a cancellable await."""

        if self._active_audio_settings_leases <= 0:
            logger.error("Audio settings lease counter underflow")
            self._active_audio_settings_leases = 0
            return
        self._active_audio_settings_leases -= 1

    @staticmethod
    def _live_audio_capture_drain_open(
        state: InterviewState, *, now: datetime | None = None
    ) -> bool:
        live = state.live_interview
        if live.status == LiveInterviewStatus.ACTIVE:
            return True
        if live.capture_closed_at is None:
            # Old paused/completed rows have no trustworthy close boundary.
            return False
        closed_at = live.capture_closed_at
        if closed_at.tzinfo is None:
            closed_at = closed_at.replace(tzinfo=timezone.utc)
        return (now or datetime.now(timezone.utc)) <= closed_at + timedelta(
            seconds=LIVE_AUDIO_CAPTURE_DRAIN_GRACE_SECONDS
        )

    def _prune_live_audio_capture_leases(
        self, state: InterviewState, *, now: datetime | None = None
    ) -> bool:
        """Discard abandoned capture registrations after the recovery window."""

        live = state.live_interview
        current_time = now or datetime.now(timezone.utc)
        cutoff = current_time - timedelta(
            seconds=LIVE_AUDIO_CAPTURE_LEASE_TTL_SECONDS
        )
        retained: list[UUID] = []
        retained_times: dict[str, datetime] = {}
        retained_etags: dict[str, str] = {}
        retained_epochs: dict[str, int] = {}
        changed = False
        guard_ids = set(live.audio_capture_guard_ids)
        drain_open = MediaServiceMixin._live_audio_capture_drain_open(
            state, now=current_time
        )
        for capture_id in live.pending_audio_capture_ids:
            key = str(capture_id)
            if (
                capture_id in guard_ids
                and live.status != LiveInterviewStatus.ACTIVE
                and not drain_open
            ):
                changed = True
                continue
            registered_at = live.pending_audio_capture_registered_at.get(key)
            # Backfill registrations written by the immediately preceding schema.
            # They get one normal recovery window instead of being invalidated at
            # upgrade time.
            if registered_at is None:
                registered_at = current_time
                changed = True
            elif registered_at.tzinfo is None:
                registered_at = registered_at.replace(tzinfo=timezone.utc)
            if registered_at < cutoff:
                changed = True
                continue
            etag = live.pending_audio_capture_settings_etags.get(key)
            epoch = live.pending_audio_capture_epochs.get(key)
            # A previous schema did not bind leases to the effective audio
            # settings or ACTIVE epoch. Such bytes cannot be processed safely;
            # retire the unusable blocker instead of leaving a zombie lease.
            if not etag or epoch is None or etag != self.audio_settings_etag:
                changed = True
                continue
            if capture_id in guard_ids and epoch != live.capture_epoch:
                changed = True
                continue
            retained.append(capture_id)
            retained_times[key] = registered_at
            retained_etags[key] = etag
            retained_epochs[key] = epoch
        if retained != live.pending_audio_capture_ids:
            live.pending_audio_capture_ids = retained
        retained_set = set(retained)
        retained_guards = [
            capture_id
            for capture_id in live.audio_capture_guard_ids
            if capture_id in retained_set
        ]
        if retained_guards != live.audio_capture_guard_ids:
            live.audio_capture_guard_ids = retained_guards
            changed = True
        if retained_times != live.pending_audio_capture_registered_at:
            live.pending_audio_capture_registered_at = retained_times
            changed = True
        if retained_etags != live.pending_audio_capture_settings_etags:
            live.pending_audio_capture_settings_etags = retained_etags
            changed = True
        if retained_epochs != live.pending_audio_capture_epochs:
            live.pending_audio_capture_epochs = retained_epochs
            changed = True
        return changed

    async def _persist_media_candidate(
        self,
        session_id: str,
        runtime: AgentRuntime,
        candidate_state: InterviewState,
        previous_state: InterviewState,
    ) -> None:
        """Commit copy-on-write media state and reconcile uncertain outcomes."""

        try:
            await self._persist(session_id, candidate_state)
        except BaseException:
            await self._recover_live_state_after_persist_error(
                session_id, runtime, previous_state
            )
            recovery_key = (current_owner(), session_id)
            if recovery_key not in self._runtime_recovery_required:
                try:
                    self._reconcile_live_audio_for_state(
                        session_id, runtime.state, strict=True
                    )
                except BaseException:  # noqa: BLE001 - retry on next lock entry
                    self._runtime_recovery_required.add(recovery_key)
            raise
        runtime.state = candidate_state

    async def ensure_audio_settings_mutable(self) -> None:
        """Reject global audio reconfiguration while recorded bytes are in flight.

        Callers hold ``_settings_lock``. Persisted capture leases cover other
        browser tabs and survive a backend restart; the in-memory lease counter
        covers live uploads, ASR previews, mock transcription, and speech-style
        analysis already being processed by this backend.
        """

        if self._active_audio_settings_leases:
            raise LiveInterviewStateError(
                "Audio settings cannot change while an audio request is processing"
            )
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(seconds=LIVE_AUDIO_CAPTURE_LEASE_TTL_SECONDS)

        def has_fresh_pending(
            pending: list[Any],
            registered: dict[str, Any],
            *,
            guards: list[Any] | None = None,
            settings_etags: dict[str, Any] | None = None,
            epochs: dict[str, Any] | None = None,
            capture_epoch: Any = None,
            status: Any = None,
            capture_closed_at: Any = None,
            legacy_registered_at: datetime | None = None,
        ) -> bool:
            guard_ids = {str(item) for item in (guards or [])}
            bound_etags = settings_etags or {}
            bound_epochs = epochs or {}
            drain_open = status == LiveInterviewStatus.ACTIVE or status == "active"
            if not drain_open and capture_closed_at is not None:
                try:
                    closed_at = (
                        capture_closed_at
                        if isinstance(capture_closed_at, datetime)
                        else datetime.fromisoformat(
                            str(capture_closed_at).replace("Z", "+00:00")
                        )
                    )
                    if closed_at.tzinfo is None:
                        closed_at = closed_at.replace(tzinfo=timezone.utc)
                    drain_open = now <= closed_at + timedelta(
                        seconds=LIVE_AUDIO_CAPTURE_DRAIN_GRACE_SECONDS
                    )
                except (TypeError, ValueError):
                    drain_open = False
            for capture_id in pending:
                key = str(capture_id)
                is_guard = key in guard_ids
                # A lease without a generation binding cannot be processed
                # safely. A lease from an older settings generation is also no
                # reason to block repair of the current configuration.
                if (
                    bound_etags.get(key) != self.audio_settings_etag
                    or key not in bound_epochs
                ):
                    continue
                if (
                    is_guard
                    and capture_epoch is not None
                    and bound_epochs.get(key) != capture_epoch
                ):
                    continue
                if is_guard and not drain_open:
                    continue
                raw_time = registered.get(key)
                if raw_time is None:
                    # Old-schema leases use the database row's update time so
                    # they expire instead of blocking settings across restarts.
                    if legacy_registered_at is None:
                        continue
                    registered_at = legacy_registered_at
                else:
                    try:
                        registered_at = (
                            raw_time
                            if isinstance(raw_time, datetime)
                            else datetime.fromisoformat(
                                str(raw_time).replace("Z", "+00:00")
                            )
                        )
                    except (TypeError, ValueError):
                        # A malformed persisted timestamp must not renew itself
                        # forever on every settings check. Bound it by the row's
                        # last durable update, or ignore an in-memory value that
                        # cannot exist in a validated runtime.
                        if legacy_registered_at is None:
                            continue
                        registered_at = legacy_registered_at
                if registered_at.tzinfo is None:
                    registered_at = registered_at.replace(tzinfo=timezone.utc)
                if registered_at >= cutoff:
                    return True
            return False

        # Inspect cache first: it can contain a registration whose SQLite write
        # failed and is waiting for an idempotent retry to heal persistence.
        for runtime in self._runtimes.values():
            live = runtime.state.live_interview
            if has_fresh_pending(
                list(live.pending_audio_capture_ids),
                dict(live.pending_audio_capture_registered_at),
                guards=list(live.audio_capture_guard_ids),
                settings_etags=dict(live.pending_audio_capture_settings_etags),
                epochs=dict(live.pending_audio_capture_epochs),
                capture_epoch=live.capture_epoch,
                status=live.status,
                capture_closed_at=live.capture_closed_at,
            ):
                raise LiveInterviewStateError(
                    "Audio settings cannot change while a recorded utterance is pending"
                )
        try:
            stored_records = await self.storage.list_all_session_state_records(
                strict=True
            )
        except (TypeError, ValueError) as exc:
            raise LiveInterviewStateError(
                "Audio settings cannot change while stored session state is invalid"
            ) from exc
        for raw_state, updated_at in stored_records:
            if not isinstance(raw_state, dict):
                raise LiveInterviewStateError(
                    "Audio settings cannot change while stored session state is invalid"
                )
            live = raw_state.get("live_interview", {})
            if not isinstance(live, dict):
                raise LiveInterviewStateError(
                    "Audio settings cannot change while stored live state is invalid"
                )
            container_shapes: tuple[tuple[str, type[Any], object], ...] = (
                ("pending_audio_capture_ids", list, []),
                ("pending_audio_capture_registered_at", dict, {}),
                ("audio_capture_guard_ids", list, []),
                ("pending_audio_capture_settings_etags", dict, {}),
                ("pending_audio_capture_epochs", dict, {}),
            )
            malformed_container = any(
                not isinstance(live.get(name, default), expected_type)
                for name, expected_type, default in container_shapes
            )
            if malformed_container:
                raise LiveInterviewStateError(
                    "Audio settings cannot change while stored live state is invalid"
                )
            if has_fresh_pending(
                live.get("pending_audio_capture_ids", []),
                live.get("pending_audio_capture_registered_at", {}),
                guards=live.get("audio_capture_guard_ids", []),
                settings_etags=live.get(
                    "pending_audio_capture_settings_etags", {}
                ),
                epochs=live.get("pending_audio_capture_epochs", {}),
                capture_epoch=live.get("capture_epoch"),
                status=live.get("status"),
                capture_closed_at=live.get("capture_closed_at"),
                legacy_registered_at=updated_at,
            ):
                raise LiveInterviewStateError(
                    "Audio settings cannot change while a recorded utterance is pending"
                )

    async def register_live_audio_capture(
        self,
        session_id: str,
        recording_id: UUID,
        *,
        expected_capture_epoch: int,
        expected_settings_revision: int | None = None,
        expected_settings_etag: str | None = None,
        parent_recording_id: UUID | None = None,
        capture_role: str = "utterance",
    ) -> InterviewState:
        """Persist proof that an utterance began before the session closed."""

        request_received_at = datetime.now(timezone.utc)
        async with self._settings_lock:
            runtime = await self._get_runtime(session_id)
            async with self._lock_for(session_id):
                previous_state = runtime.state
                candidate_state = runtime.state.model_copy(deep=True)
                live = candidate_state.live_interview
                changed = self._prune_live_audio_capture_leases(
                    candidate_state, now=request_received_at
                )
                if changed:
                    await self._persist_media_candidate(
                        session_id, runtime, candidate_state, previous_state
                    )
                    previous_state = runtime.state
                    candidate_state = runtime.state.model_copy(deep=True)
                    live = candidate_state.live_interview
                existing = next(
                    (item for item in live.audio_capture_receipts if item.id == recording_id),
                    None,
                )
                if existing is not None:
                    changed = False
                    if recording_id in live.pending_audio_capture_ids:
                        live.pending_audio_capture_ids.remove(recording_id)
                        changed = True
                    if live.pending_audio_capture_registered_at.pop(str(recording_id), None):
                        changed = True
                    if live.pending_audio_capture_settings_etags.pop(str(recording_id), None):
                        changed = True
                    if live.pending_audio_capture_epochs.pop(str(recording_id), None) is not None:
                        changed = True
                    if recording_id in live.audio_capture_guard_ids:
                        live.audio_capture_guard_ids.remove(recording_id)
                        changed = True
                    if changed:
                        await self._persist_media_candidate(
                            session_id, runtime, candidate_state, previous_state
                        )
                    return runtime.state
                if (
                    expected_settings_etag is not None
                    and expected_settings_etag != self.audio_settings_etag
                ):
                    raise LiveInterviewStateError(
                        "Audio settings changed; reload settings before recording"
                    )
                if (
                    expected_settings_revision is not None
                    and expected_settings_revision != self.audio_settings_revision
                ):
                    raise LiveInterviewStateError(
                        "Audio settings changed; reload settings before recording"
                    )
                if recording_id in live.cancelled_audio_capture_ids:
                    raise LiveInterviewStateError("Audio capture was already cancelled")
                if capture_role not in {"utterance", "guard"}:
                    raise LiveInterviewStateError("Unsupported audio capture role")
                if recording_id in live.pending_audio_capture_ids:
                    # The first persistence attempt may have failed after the
                    # in-memory append. Always save on an idempotent retry to
                    # heal that uncertain outcome. It also renews an active
                    # browser's lease so the TTL only removes abandoned audio.
                    registered_etag = live.pending_audio_capture_settings_etags.get(
                        str(recording_id)
                    )
                    if not registered_etag or registered_etag != self.audio_settings_etag:
                        raise LiveInterviewStateError(
                            "Audio settings changed; capture belongs to an earlier generation"
                        )
                    if (
                        live.pending_audio_capture_epochs.get(str(recording_id))
                        != expected_capture_epoch
                    ):
                        raise LiveInterviewStateError(
                            "Live audio capture epoch changed; restart recording"
                        )
                    is_guard = recording_id in live.audio_capture_guard_ids
                    if is_guard != (capture_role == "guard"):
                        raise LiveInterviewStateError(
                            "Audio capture role does not match its registration"
                        )
                    if live.status == LiveInterviewStatus.ACTIVE or (
                        not is_guard
                        and live.status
                        in {
                            LiveInterviewStatus.PAUSED,
                            LiveInterviewStatus.COMPLETED,
                        }
                    ):
                        # A tab that still holds the finalized bytes may keep an
                        # utterance retryable while the interview is closed.
                        # Guards never renew after close and therefore cannot
                        # authorize new child captures indefinitely.
                        live.pending_audio_capture_registered_at[str(recording_id)] = (
                            request_received_at
                        )
                    await self._persist_media_candidate(
                        session_id, runtime, candidate_state, previous_state
                    )
                    return runtime.state
                if expected_capture_epoch != live.capture_epoch:
                    raise LiveInterviewStateError(
                        "Live audio capture epoch changed; restart recording"
                    )
                parent_is_pending = bool(
                    parent_recording_id
                    and parent_recording_id != recording_id
                    and parent_recording_id in live.pending_audio_capture_ids
                    and parent_recording_id in live.audio_capture_guard_ids
                    and live.pending_audio_capture_settings_etags.get(
                        str(parent_recording_id)
                    )
                    == self.audio_settings_etag
                    and live.pending_audio_capture_epochs.get(str(parent_recording_id))
                    == live.capture_epoch
                )
                parent_authorizes_drain = (
                    capture_role == "utterance"
                    and parent_is_pending
                    and live.status
                    in {LiveInterviewStatus.PAUSED, LiveInterviewStatus.COMPLETED}
                    and self._live_audio_capture_drain_open(
                        candidate_state, now=request_received_at
                    )
                )
                if (
                    live.status != LiveInterviewStatus.ACTIVE
                    and not parent_authorizes_drain
                ):
                    raise LiveInterviewStateError(
                        "Live interview must be active before registering audio capture"
                    )
                if len(live.pending_audio_capture_ids) >= LIVE_AUDIO_PENDING_CAPTURE_MAX:
                    raise LiveInterviewStateError(
                        "Too many pending audio captures; save pending audio first"
                    )
                live.pending_audio_capture_ids.append(recording_id)
                if capture_role == "guard":
                    live.audio_capture_guard_ids.append(recording_id)
                live.pending_audio_capture_registered_at[str(recording_id)] = (
                    request_received_at
                )
                live.pending_audio_capture_settings_etags[str(recording_id)] = (
                    self.audio_settings_etag
                )
                live.pending_audio_capture_epochs[str(recording_id)] = live.capture_epoch
                await self._persist_media_candidate(
                    session_id, runtime, candidate_state, previous_state
                )
        self._record_debug("live_audio_capture_registered", session_id)
        return runtime.state

    async def cancel_live_audio_capture(
        self, session_id: str, recording_id: UUID
    ) -> InterviewState:
        """Release a registration when the browser produced no uploadable audio."""

        async with self._settings_lock:
            runtime = await self._get_runtime(session_id)
            async with self._lock_for(session_id):
                previous_state = runtime.state
                candidate_state = runtime.state.model_copy(deep=True)
                live = candidate_state.live_interview
                changed = self._prune_live_audio_capture_leases(candidate_state)
                if recording_id in live.pending_audio_capture_ids:
                    live.pending_audio_capture_ids.remove(recording_id)
                    changed = True
                if live.pending_audio_capture_registered_at.pop(str(recording_id), None):
                    changed = True
                if live.pending_audio_capture_settings_etags.pop(str(recording_id), None):
                    changed = True
                if live.pending_audio_capture_epochs.pop(str(recording_id), None) is not None:
                    changed = True
                if recording_id in live.audio_capture_guard_ids:
                    live.audio_capture_guard_ids.remove(recording_id)
                    changed = True
                if (
                    recording_id not in live.cancelled_audio_capture_ids
                    and not any(
                        item.id == recording_id for item in live.audio_capture_receipts
                    )
                ):
                    live.cancelled_audio_capture_ids.append(recording_id)
                    if (
                        len(live.cancelled_audio_capture_ids)
                        > LIVE_AUDIO_CANCELLED_CAPTURE_MAX
                    ):
                        del live.cancelled_audio_capture_ids[
                            :-LIVE_AUDIO_CANCELLED_CAPTURE_MAX
                        ]
                    changed = True
                # Persist even when already cancelled so a retry after an
                # uncertain response still receives a durable terminal state.
                await self._persist_media_candidate(
                    session_id, runtime, candidate_state, previous_state
                )
        if changed:
            self._record_debug("live_audio_capture_cancelled", session_id)
        return runtime.state

    async def transcribe_live_audio(
        self,
        session_id: str,
        *,
        content: bytes,
        filename: str,
        content_type: str,
        speaker: TranscriptSpeaker,
        language: str = "zh",
        mode: str = "single",
        recording_id: UUID | None = None,
    ) -> tuple[InterviewState, str]:
        if len(content) > 25 * 1024 * 1024:
            raise LiveInterviewStateError("Audio chunk exceeds the 25 MB limit")
        if not content:
            raise LiveInterviewStateError("Audio chunk is empty")
        runtime = await self._get_runtime(session_id)
        if recording_id is None:
            await self._acquire_audio_settings_lease()
            live_audio_mode = self.live_audio_mode
            try:
                return await self._transcribe_live_audio_once(
                    session_id,
                    runtime,
                    content=content,
                    filename=filename,
                    content_type=content_type,
                    speaker=speaker,
                    language=language,
                    mode=mode,
                    live_audio_mode=live_audio_mode,
                    receipt=None,
                )
            finally:
                # This counter is event-loop local. A cancellable await here
                # could leak it and block every future settings update.
                self._release_audio_settings_lease()

        fingerprint = self._live_audio_fingerprint(
            content,
            filename=filename,
            content_type=content_type,
            speaker=speaker,
            language=language,
            mode=mode,
        )
        key = (current_owner(), session_id, recording_id)
        in_flight = self._live_audio_in_flight.get(key)
        if in_flight is not None:
            prior_fingerprint, task = in_flight
            if prior_fingerprint != fingerprint:
                raise LiveInterviewStateError(
                    "语音片段标识已用于不同内容，已拒绝重复处理"
                )
            return await asyncio.shield(task)

        # Acquire the lease before publishing work so settings cannot change in
        # the small interval between accepting the audio and starting the task.
        await self._acquire_audio_settings_lease()
        in_flight = self._live_audio_in_flight.get(key)
        if in_flight is not None:
            self._release_audio_settings_lease()
            prior_fingerprint, task = in_flight
            if prior_fingerprint != fingerprint:
                raise LiveInterviewStateError(
                    "语音片段标识已用于不同内容，已拒绝重复处理"
                )
            return await asyncio.shield(task)

        live_audio_mode = self.live_audio_mode
        try:
            task = asyncio.create_task(
                self._transcribe_live_audio_with_owned_lease(
                    session_id,
                    runtime,
                    content=content,
                    filename=filename,
                    content_type=content_type,
                    speaker=speaker,
                    language=language,
                    mode=mode,
                    live_audio_mode=live_audio_mode,
                    receipt=LiveAudioCaptureReceipt(
                        id=recording_id,
                        payload_sha256=fingerprint,
                    ),
                )
            )
        except BaseException:
            self._release_audio_settings_lease()
            raise
        self._live_audio_in_flight[key] = (fingerprint, task)
        task.add_done_callback(partial(self._finish_live_audio_task, key))
        return await asyncio.shield(task)

    async def _transcribe_live_audio_with_owned_lease(
        self,
        session_id: str,
        runtime: AgentRuntime,
        **values: Any,
    ) -> tuple[InterviewState, str]:
        """Run shared audio work whose settings lease belongs to this task."""

        try:
            return await self._transcribe_live_audio_once(session_id, runtime, **values)
        finally:
            self._release_audio_settings_lease()

    def _finish_live_audio_task(
        self,
        key: tuple[str, str, UUID],
        task: asyncio.Future[tuple[InterviewState, str]],
    ) -> None:
        """Drop completed single-flight work and consume orphaned failures."""

        current = self._live_audio_in_flight.get(key)
        if current is not None and current[1] is task:
            self._live_audio_in_flight.pop(key, None)
        if task.cancelled():
            return
        try:
            task.exception()
        except asyncio.CancelledError:
            return

    def _live_audio_fingerprint(
        self,
        content: bytes,
        *,
        filename: str,
        content_type: str,
        speaker: TranscriptSpeaker,
        language: str,
        mode: str,
    ) -> str:
        digest = hashlib.sha256()
        digest.update(content)
        for value in (
            filename,
            content_type,
            speaker.value,
            language,
            mode,
        ):
            digest.update(b"\0")
            digest.update(value.encode("utf-8"))
        return digest.hexdigest()

    async def _transcribe_live_audio_once(
        self,
        session_id: str,
        runtime: AgentRuntime,
        *,
        content: bytes,
        filename: str,
        content_type: str,
        speaker: TranscriptSpeaker,
        language: str,
        mode: str,
        live_audio_mode: str,
        receipt: LiveAudioCaptureReceipt | None,
    ) -> tuple[InterviewState, str]:
        async with self._lock_for(session_id):
            previous_state = runtime.state
            candidate_state = runtime.state.model_copy(deep=True)
            live = candidate_state.live_interview
            lease_changed = self._prune_live_audio_capture_leases(candidate_state)
            if lease_changed:
                # Pruning is an independent durable mutation. Commit it before
                # receipt validation can return or raise; otherwise expired
                # blockers could reappear after a restart.
                await self._persist_media_candidate(
                    session_id, runtime, candidate_state, previous_state
                )
                previous_state = runtime.state
                candidate_state = runtime.state.model_copy(deep=True)
                live = candidate_state.live_interview
                lease_changed = False
            if receipt is not None:
                existing = next(
                    (
                        item
                        for item in live.audio_capture_receipts
                        if item.id == receipt.id
                    ),
                    None,
                )
                if existing is not None:
                    if existing.payload_sha256 != receipt.payload_sha256:
                        raise LiveInterviewStateError(
                            "语音片段标识已用于不同内容，已拒绝重复处理"
                        )
                    if receipt.id in live.pending_audio_capture_ids:
                        live.pending_audio_capture_ids.remove(receipt.id)
                        lease_changed = True
                    if live.pending_audio_capture_registered_at.pop(str(receipt.id), None):
                        lease_changed = True
                    if live.pending_audio_capture_settings_etags.pop(str(receipt.id), None):
                        lease_changed = True
                    if live.pending_audio_capture_epochs.pop(str(receipt.id), None) is not None:
                        lease_changed = True
                    if lease_changed:
                        await self._persist_media_candidate(
                            session_id, runtime, candidate_state, previous_state
                        )
                    return runtime.state, ""
                if receipt.id in live.cancelled_audio_capture_ids:
                    raise LiveInterviewStateError("Audio capture was already cancelled")
                if receipt.id in live.audio_capture_guard_ids:
                    raise LiveInterviewStateError("Audio capture guard cannot contain audio")
            registered_before_stop = bool(
                receipt and receipt.id in live.pending_audio_capture_ids
            )
            if receipt is not None and registered_before_stop and (
                live.pending_audio_capture_settings_etags.get(str(receipt.id))
                != self.audio_settings_etag
            ):
                raise LiveInterviewStateError(
                    "语音片段属于旧的音频配置；请重新录制或放弃该片段"
                )
            valid_new_capture = (
                live.status == LiveInterviewStatus.ACTIVE and receipt is None
            ) or (
                registered_before_stop
                and live.status
                in {
                    LiveInterviewStatus.ACTIVE,
                    LiveInterviewStatus.PAUSED,
                    LiveInterviewStatus.COMPLETED,
                }
            )
            if not valid_new_capture:
                raise LiveInterviewStateError(
                    "Live interview must be active, or audio must have been registered before it stopped"
                )
        started = perf_counter()
        if live_audio_mode == "audio_direct":
            if self.omni_client is None:
                raise LiveInterviewStateError("Omni audio client is not configured")
            if mode == "dialogue":
                return await self._transcribe_dialogue(
                    session_id,
                    runtime,
                    content,
                    content_type=content_type,
                    started=started,
                    receipt=receipt,
                )
            if speaker != TranscriptSpeaker.CANDIDATE:
                raise LiveInterviewStateError("音频直连模式仅支持候选人回答")
            context = self._live_audio_context(runtime.state)
            result = await self.omni_client.suggest_next_question(
                content, content_type=content_type, context=context, stream=False
            )
            suggestion_text = result if isinstance(result, str) else ""
            suggestion = self._build_audio_direct_suggestion(runtime, suggestion_text)
            await self._commit_live_audio_capture(
                session_id,
                runtime,
                suggestion=suggestion,
                receipt=receipt,
            )
            self._record_debug(
                "audio_direct_suggestion",
                session_id,
                detail=f"chars={len(suggestion_text)}",
                duration_ms=(perf_counter() - started) * 1000,
            )
            return runtime.state, suggestion_text
        if self.asr_client is None:
            raise LiveInterviewStateError("ASR client is not configured")
        try:
            transcript = await self.asr_client.transcribe(
                content,
                filename=filename,
                content_type=content_type,
                language=language,
            )
        except ASRError as exc:
            self._record_debug(
                "asr_transcription_failed",
                session_id,
                level=DebugLevel.ERROR,
                detail=f"error_type={type(exc).__name__}",
                duration_ms=(perf_counter() - started) * 1000,
            )
            raise WorkflowExecutionError("Audio transcription failed") from exc
        state = await self._commit_live_audio_capture(
            session_id,
            runtime,
            segments=[(speaker, transcript, "asr")],
            receipt=receipt,
        )
        self._record_debug(
            "asr_transcription_completed",
            session_id,
            detail=f"speaker={speaker.value}; bytes={len(content)}; chars={len(transcript)}",
            duration_ms=(perf_counter() - started) * 1000,
        )
        return state, transcript

    async def preview_audio_transcription(
        self,
        session_id: str,
        *,
        content: bytes,
        filename: str,
        content_type: str,
        language: str = "zh",
    ) -> str:
        """Transcribe a cumulative recording snapshot without persisting it.

        The browser calls this at a bounded interval while recording so users
        can see provisional words. Only the final audio request creates a stable
        transcript segment, preventing duplicate evidence and partial text from
        entering downstream agents.
        """
        await self._get_runtime(session_id)
        if len(content) > 10 * 1024 * 1024:
            raise LiveInterviewStateError("ASR preview exceeds the 10 MB limit")
        if not content:
            raise LiveInterviewStateError("ASR preview audio is empty")
        await self._acquire_audio_settings_lease()
        asr_client = self.asr_client
        started = perf_counter()
        try:
            if asr_client is None:
                raise LiveInterviewStateError("ASR client is not configured")
            try:
                transcript = await asr_client.transcribe(
                    content,
                    filename=filename,
                    content_type=content_type,
                    language=language,
                )
            except ASRError as exc:
                self._record_debug(
                    "asr_preview_failed",
                    session_id,
                    level=DebugLevel.ERROR,
                    detail=f"error_type={type(exc).__name__}",
                    duration_ms=(perf_counter() - started) * 1000,
                )
                raise WorkflowExecutionError("Audio transcription failed") from exc
        finally:
            self._release_audio_settings_lease()
        self._record_debug(
            "asr_preview_completed",
            session_id,
            detail=f"bytes={len(content)}; chars={len(transcript)}",
            duration_ms=(perf_counter() - started) * 1000,
        )
        return transcript

    async def transcribe_mock_spoken_answer(
        self, session_id: str, content: bytes, filename: str
    ) -> str:
        """Transcribe a spoken mock-interview answer to text.

        A pure transcription: it does not touch live state, does not persist the
        audio, and does not change mock state — the caller fills the answer into
        the mock answer field. Ownership is enforced by ``_get_runtime``.
        """
        await self._get_runtime(session_id)
        content_type = (
            "audio/wav"
            if filename.lower().endswith(".wav")
            else "audio/webm"
            if filename.lower().endswith(".webm")
            else "audio/mp4"
            if filename.lower().endswith(".m4a")
            else "application/octet-stream"
        )
        await self._acquire_audio_settings_lease()
        asr_client = self.asr_client
        started = perf_counter()
        try:
            if asr_client is None:
                raise WorkflowExecutionError("ASR client is not configured")
            try:
                transcript = await asr_client.transcribe(
                    content,
                    filename=filename,
                    content_type=content_type,
                    language="zh",
                )
            except ASRError as exc:
                self._record_debug(
                    "mock_asr_transcription_failed",
                    session_id,
                    level=DebugLevel.ERROR,
                    detail=f"error_type={type(exc).__name__}",
                    duration_ms=(perf_counter() - started) * 1000,
                )
                raise WorkflowExecutionError("Audio transcription failed") from exc
        finally:
            self._release_audio_settings_lease()
        self._record_debug(
            "mock_asr_transcription_completed",
            session_id,
            detail=f"bytes={len(content)}; chars={len(transcript)}",
            duration_ms=(perf_counter() - started) * 1000,
        )
        return transcript

    async def save_mock_answer_audio(
        self,
        session_id: str,
        recording_id: UUID,
        content: bytes,
        *,
        extension: str = "wav",
        question_id: UUID | None = None,
        question: str = "",
        retry: bool = False,
        retry_response_id: UUID | None = None,
    ) -> str:
        """Persist and index a recoverable draft before attempting transcription."""
        runtime = await self._get_runtime(session_id)
        ext = extension.lstrip(".").lower()
        if ext not in {"wav", "webm", "m4a"}:
            ext = "wav"
        filename = f"mock-{session_id}-{recording_id}.{ext}"
        async with self._lock_for(session_id):
            current = self.current_mock_question(runtime.state)
            if current is None:
                raise MockInterviewStateError("Mock interview is not active")
            expected_id = runtime.state.mock_session.pending_parent_question_id or current.id
            resolved_question_id = question_id or expected_id
            if resolved_question_id != expected_id:
                raise MockInterviewStateError("Recording does not match the current question")
            # Preserve the exact wording: submit_mock_answer binds a draft to
            # the concrete main/follow-up prompt, including intentional line
            # breaks or repeated spaces.
            resolved_question = (question or current.question).strip()
            if not resolved_question or len(resolved_question) > 2000:
                raise MockInterviewStateError("Recording question is invalid")
            previous_draft = runtime.state.mock_session.answer_draft
            self._ensure_private_recordings_dir()
            target = self._recordings_dir / filename
            temporary = target.with_suffix(f".{ext}.tmp")
            self._write_then_replace(temporary, target, content)
            runtime.state.mock_session.answer_draft = MockAnswerDraft(
                recording_id=recording_id,
                question_id=resolved_question_id,
                question=resolved_question,
                audio_file=filename,
                retry=retry,
                retry_response_id=retry_response_id if retry else None,
            )
            await self._persist(session_id, runtime.state)
            if previous_draft is not None and previous_draft.audio_file != filename:
                old_path = self.get_mock_answer_audio_path(
                    session_id, previous_draft.audio_file
                )
                if old_path is not None:
                    old_path.unlink(missing_ok=True)
            referenced = {
                item.audio_file
                for item in (
                    *runtime.state.mock_session.responses,
                    *runtime.state.mock_session.attempt_history,
                )
                if item.audio_file
            }
            referenced.add(filename)
            cutoff = datetime.now(timezone.utc).timestamp() - 86400
            for candidate in self._recordings_dir.glob(f"mock-{session_id}-*"):
                try:
                    if (
                        candidate.name not in referenced
                        and candidate != target
                        and candidate.stat().st_mtime < cutoff
                    ):
                        candidate.unlink(missing_ok=True)
                except OSError:
                    continue
        self._record_debug(
            "mock_answer_audio_staged",
            session_id,
            detail=f"recording_id={recording_id}; bytes={len(content)}",
        )
        return filename

    async def update_mock_answer_draft(
        self,
        session_id: str,
        recording_id: UUID,
        *,
        transcript: str,
        status: str,
        transcription_error: str = "",
        speech_delivery: SpeechDeliveryFeedback | None = None,
    ) -> MockAnswerDraft:
        """Attach final ASR/coaching output to the still-unsubmitted draft."""

        if status not in {"saved", "completed", "failed"}:
            raise ValueError("Unsupported mock answer draft status")
        runtime = await self._get_runtime(session_id)
        async with self._lock_for(session_id):
            draft = runtime.state.mock_session.answer_draft
            if draft is None or draft.recording_id != recording_id:
                raise MockInterviewStateError("Mock answer draft was replaced")
            draft.transcript = transcript.strip()
            draft.transcription_status = status
            draft.transcription_error = transcription_error
            draft.updated_at = datetime.now(timezone.utc)
            if speech_delivery is not None:
                cached = self._mock_speech_feedback.get(
                    (current_owner(), session_id, recording_id)
                )
                # A very fast audio model may finish between scheduling and
                # this persistence call. Never overwrite richer completed
                # feedback with the immediate deterministic fallback.
                draft.speech_delivery = (
                    cached[1]
                    if cached is not None and cached[0] == "completed"
                    else speech_delivery
                )
            await self._persist(session_id, runtime.state)
            return draft.model_copy(deep=True)

    def _pending_mock_audio_path(self, session_id: str, recording_id: UUID) -> Path | None:
        for ext in ("wav", "webm", "m4a"):
            candidate = self._recordings_dir / f"mock-{session_id}-{recording_id}.{ext}"
            if candidate.is_file():
                return candidate
        return None

    def get_mock_answer_audio_path(self, session_id: str, stored_filename: str) -> Path | None:
        pattern = re.compile(rf"^mock-{re.escape(session_id)}-[0-9a-f-]{{36}}\.(?:wav|webm|m4a)$")
        if not pattern.fullmatch(stored_filename):
            return None
        candidate = self._recordings_dir / stored_filename
        return candidate if candidate.is_file() else None

    async def cleanup_orphaned_mock_audio(self, *, max_age_seconds: int = 86400) -> int:
        """Remove expired staged files while retaining every persisted attempt."""
        referenced: set[str] = set()
        try:
            states = [
                InterviewState.model_validate(state)
                for state in await self.storage.list_all_session_states(strict=True)
            ]
        except (TypeError, ValueError):
            logger.error(
                "Skipped mock audio cleanup because the session snapshot is incomplete"
            )
            return 0
        for state in states:
            mock = state.mock_session.model_dump(mode="json")
            for bucket in ("responses", "attempt_history"):
                records = mock.get(bucket)
                if not isinstance(records, list):
                    continue
                referenced.update(
                    str(item.get("audio_file"))
                    for item in records
                    if isinstance(item, dict) and item.get("audio_file")
                )
            draft = mock.get("answer_draft")
            if isinstance(draft, dict) and draft.get("audio_file"):
                referenced.add(str(draft["audio_file"]))
        cutoff = datetime.now(timezone.utc).timestamp() - max(0, max_age_seconds)
        removed = 0
        if not self._recordings_dir.exists():
            return removed
        for candidate in self._recordings_dir.glob("mock-*"):
            try:
                if candidate.name not in referenced and candidate.stat().st_mtime < cutoff:
                    candidate.unlink(missing_ok=True)
                    removed += 1
            except OSError:
                continue
        return removed

    @staticmethod
    def _live_audio_references(live: Any) -> dict[str, str]:
        """Return the filenames and payload hashes authorized by one manifest."""

        if hasattr(live, "model_dump"):
            live = live.model_dump(mode="json")
        if not isinstance(live, dict):
            return {}
        references: dict[str, str] = {}
        for part in live.get("audio_parts", []):
            if isinstance(part, dict) and part.get("audio_file"):
                references[str(part["audio_file"])] = str(
                    part.get("payload_sha256") or ""
                )
        if live.get("audio_file"):
            references[str(live["audio_file"])] = str(
                live.get("audio_payload_sha256") or ""
            )
        return references

    @staticmethod
    def _live_audio_name_matches_session(name: str, session_id: str | None) -> bool:
        if session_id is None:
            return True
        return bool(
            re.fullmatch(
                rf"{re.escape(session_id)}(?:-[0-9a-f-]{{36}})?\.(?:wav|webm|m4a)",
                name,
            )
        )

    def _reconcile_live_audio_files(
        self,
        references: dict[str, str],
        *,
        session_id: str | None = None,
        strict: bool = False,
    ) -> tuple[int, int]:
        """Converge audio journals and files onto an authoritative manifest."""

        if not self._recordings_dir.exists():
            return 0, 0
        recovered = 0
        removed = 0
        failed = False
        delete_pattern = re.compile(
            r"^\.(?P<original>.+\.(?:wav|webm|m4a))\.[0-9a-f]{32}\.delete-pending$"
        )
        replace_pattern = re.compile(
            r"^\.(?P<original>.+\.(?:wav|webm|m4a))\.[0-9a-f]{32}\.replace-pending$"
        )

        for quarantine in self._recordings_dir.glob(".*.delete-pending"):
            match = delete_pattern.fullmatch(quarantine.name)
            if match is None:
                continue
            original_name = match.group("original")
            if not self._live_audio_name_matches_session(original_name, session_id):
                continue
            original = self._recordings_dir / original_name
            try:
                if original_name in references and not original.exists():
                    os.replace(quarantine, original)
                    recovered += 1
                else:
                    quarantine.unlink(missing_ok=True)
                    removed += 1
            except OSError:
                failed = True
                logger.warning("Could not reconcile recording quarantine %s", quarantine.name)

        for backup in self._recordings_dir.glob(".*.replace-pending"):
            match = replace_pattern.fullmatch(backup.name)
            if match is None:
                continue
            original_name = match.group("original")
            if not self._live_audio_name_matches_session(original_name, session_id):
                continue
            original = self._recordings_dir / original_name
            expected_hash = references.get(original_name)
            try:
                original_matches = bool(
                    expected_hash
                    and original.is_file()
                    and self._file_sha256(original) == expected_hash
                )
                backup_matches = bool(
                    expected_hash and self._file_sha256(backup) == expected_hash
                )
                if original_matches or original_name not in references:
                    backup.unlink(missing_ok=True)
                    removed += 1
                elif not expected_hash or backup_matches:
                    os.replace(backup, original)
                    recovered += 1
                else:
                    failed = True
                    logger.error(
                        "Recording backup does not match persisted manifest: %s",
                        backup.name,
                    )
            except OSError:
                failed = True
                logger.warning("Could not reconcile recording backup %s", backup.name)

        live_name = re.compile(
            r"^[0-9a-f-]{36}(?:-[0-9a-f-]{36})?\.(?:wav|webm|m4a)$"
        )
        for candidate in self._recordings_dir.iterdir():
            if not candidate.is_file() or not live_name.fullmatch(candidate.name):
                continue
            if not self._live_audio_name_matches_session(candidate.name, session_id):
                continue
            if candidate.name in references:
                continue
            try:
                candidate.unlink(missing_ok=True)
                removed += 1
            except OSError:
                failed = True
                logger.warning("Could not remove orphaned live recording %s", candidate.name)
        for temporary in self._recordings_dir.glob("*.tmp"):
            original_name = temporary.name.removesuffix(".tmp")
            if not self._live_audio_name_matches_session(original_name, session_id):
                continue
            try:
                temporary.unlink(missing_ok=True)
                removed += 1
            except OSError:
                failed = True
        if strict and failed:
            raise LiveInterviewStateError(
                "录音文件事务尚未恢复完成；请检查存储权限后重试"
            )
        return recovered, removed

    def _reconcile_live_audio_for_state(
        self,
        session_id: str,
        state: InterviewState,
        *,
        strict: bool = False,
    ) -> tuple[int, int]:
        return self._reconcile_live_audio_files(
            self._live_audio_references(state.live_interview),
            session_id=session_id,
            strict=strict,
        )

    async def cleanup_orphaned_live_audio(self) -> tuple[int, int]:
        """Reconcile crash leftovers from whole-session audio transactions.

        This startup pass is global because no upload transaction can yet be
        active. Runtime uncertainty uses the same reconciler scoped to one
        locked session.
        """

        references: dict[str, str] = {}
        try:
            states = [
                InterviewState.model_validate(state)
                for state in await self.storage.list_all_session_states(strict=True)
            ]
        except (TypeError, ValueError):
            logger.error(
                "Skipped live audio cleanup because the session snapshot is incomplete"
            )
            return 0, 0
        for state in states:
            references.update(self._live_audio_references(state.live_interview))
        return self._reconcile_live_audio_files(references)

    async def delete_mock_answer_audio(self, session_id: str, response_id: UUID) -> bool:
        runtime = await self._get_runtime(session_id)
        filename = ""
        async with self._lock_for(session_id):
            records = [
                *runtime.state.mock_session.responses,
                *runtime.state.mock_session.attempt_history,
            ]
            record = next((item for item in records if item.id == response_id), None)
            if record is None:
                raise MockInterviewStateError("Mock answer was not found")
            filename = record.audio_file
            if not filename:
                return False
            path = self.get_mock_answer_audio_path(session_id, filename)
            if path is not None:
                try:
                    path.unlink(missing_ok=True)
                except OSError as exc:
                    raise WorkflowExecutionError(
                        "Unable to delete the local mock recording"
                    ) from exc
            record.audio_file = ""
            await self._persist(session_id, runtime.state)
        return True

    async def discard_mock_answer_draft(
        self, session_id: str, recording_id: UUID
    ) -> bool:
        """Delete one recoverable draft and its local audio by explicit user action."""

        runtime = await self._get_runtime(session_id)
        async with self._lock_for(session_id):
            draft = runtime.state.mock_session.answer_draft
            if draft is None or draft.recording_id != recording_id:
                raise MockInterviewStateError("Mock answer draft was not found")
            path = self.get_mock_answer_audio_path(session_id, draft.audio_file)
            if path is not None:
                try:
                    path.unlink(missing_ok=True)
                except OSError as exc:
                    raise WorkflowExecutionError(
                        "Unable to delete the local mock answer draft"
                    ) from exc
            runtime.state.mock_session.answer_draft = None
            self._mock_speech_feedback.pop((current_owner(), session_id, recording_id), None)
            await self._persist(session_id, runtime.state)
        return True

    async def analyze_mock_speech_delivery(
        self,
        session_id: str,
        content: bytes,
        transcript: str,
        *,
        content_type: str = "audio/wav",
    ) -> SpeechDeliveryFeedback:
        """Return coaching-only delivery feedback with a safe local fallback."""
        await self._get_runtime(session_id)
        await self._acquire_audio_settings_lease()
        omni_client = self.omni_client
        try:
            return await self._analyze_mock_speech_delivery_unleased(
                session_id,
                content,
                transcript,
                content_type=content_type,
                omni_client=omni_client,
            )
        finally:
            self._release_audio_settings_lease()

    async def _analyze_mock_speech_delivery_unleased(
        self,
        session_id: str,
        content: bytes,
        transcript: str,
        *,
        content_type: str,
        omni_client: Any,
    ) -> SpeechDeliveryFeedback:
        """Analyze delivery while the caller owns an audio-settings lease."""

        fallback = self._build_mock_speech_fallback(content, transcript, content_type)
        duration = self._wav_duration_seconds(content) if "wav" in content_type else 0.0
        clean = transcript.strip()
        if omni_client is None or not hasattr(omni_client, "analyze_speaking_style"):
            return fallback
        started = perf_counter()
        try:
            result = await asyncio.wait_for(
                omni_client.analyze_speaking_style(
                    content,
                    transcript=clean,
                    content_type=content_type,
                ),
                timeout=25.0,
            )
        except TimeoutError:
            self._record_debug(
                "mock_speech_delivery_fallback",
                session_id,
                detail="reason=audio_model_timeout; source=text_fallback",
                duration_ms=(perf_counter() - started) * 1000,
            )
            return fallback
        except Exception as exc:  # noqa: BLE001 - provider output must safely degrade
            self._record_debug(
                "mock_speech_delivery_fallback",
                session_id,
                detail=f"reason=audio_model_error; type={type(exc).__name__}",
                duration_ms=(perf_counter() - started) * 1000,
            )
            return fallback
        allowed_text = ("pace", "pauses", "fillers", "volume", "intonation", "clarity")
        if not isinstance(result, dict) or not any(
            str(result.get(key) or "").strip() for key in allowed_text
        ):
            return fallback
        if not self._speech_feedback_is_compliant(result):
            self._record_debug(
                "mock_speech_delivery_fallback",
                session_id,
                detail="reason=prohibited_inference; source=text_fallback",
                duration_ms=(perf_counter() - started) * 1000,
            )
            return fallback
        values = {key: str(result.get(key) or getattr(fallback, key))[:500] for key in allowed_text}
        raw_strengths = result.get("strengths")
        raw_improvements = result.get("improvements")
        feedback = SpeechDeliveryFeedback(
            source="audio_model",
            **values,
            strengths=[str(item)[:300] for item in raw_strengths[:4]]
            if isinstance(raw_strengths, list)
            else fallback.strengths,
            improvements=[str(item)[:300] for item in raw_improvements[:4]]
            if isinstance(raw_improvements, list)
            else fallback.improvements,
        )
        self._record_debug(
            "mock_speech_delivery_analyzed",
            session_id,
            detail=f"source=audio_model; duration_seconds={duration:.1f}",
            duration_ms=(perf_counter() - started) * 1000,
        )
        return feedback

    def _build_mock_speech_fallback(
        self, content: bytes, transcript: str, content_type: str
    ) -> SpeechDeliveryFeedback:
        duration = self._wav_duration_seconds(content) if "wav" in content_type else 0.0
        clean = transcript.strip()
        filler_count = sum(clean.count(word) for word in ("嗯", "啊", "然后", "就是", "那个"))
        chars_per_minute = round(len(clean) * 60 / duration) if duration > 0 else 0
        pace = (
            f"约 {chars_per_minute} 字/分钟，语速偏快，关键结论后可停顿 1–2 秒。"
            if chars_per_minute > 300
            else f"约 {chars_per_minute} 字/分钟，语速偏慢，可先说结论再补证据。"
            if 0 < chars_per_minute < 120
            else f"约 {chars_per_minute} 字/分钟，处于易跟随区间。"
            if chars_per_minute
            else "未获得可靠音频时长，无法计算语速。"
        )
        return SpeechDeliveryFeedback(
            pace=pace,
            fillers=(
                f"转写中识别到约 {filler_count} 处常见填充词，可用短停顿替代。"
                if filler_count
                else "转写中未识别到明显填充词。"
            ),
            clarity=(
                "转写文本过短，暂时无法判断表达清晰度。"
                if len(clean) < 20
                else "转写文本可读；请用“结论—行动—结果”的句式进一步提高清晰度。"
            ),
            strengths=["已完成可回放的口语回答，可对照转写文本自查。"],
            improvements=[
                "回放时只检查一个目标：每个关键结论后留出 1–2 秒停顿。",
                "下次开头先用一句话给出结论，再说行动和可量化结果。",
            ],
        )

    async def schedule_mock_speech_delivery_analysis(
        self,
        session_id: str,
        recording_id: UUID,
        content: bytes,
        transcript: str,
        *,
        content_type: str,
    ) -> SpeechDeliveryFeedback:
        """Return text feedback immediately and enrich it from audio in background."""
        await self._get_runtime(session_id)
        owner = current_owner()
        now = datetime.now(timezone.utc).timestamp()
        expired = [
            key
            for key, (_, _, created_at) in self._mock_speech_feedback.items()
            if created_at < now - 3600
        ]
        for key in expired:
            self._mock_speech_feedback.pop(key, None)
        fallback = self._build_mock_speech_fallback(content, transcript, content_type)
        key = (owner, session_id, recording_id)
        generation = await self._acquire_audio_settings_lease()
        omni_client = self.omni_client
        self._mock_speech_feedback[key] = ("analyzing", fallback, now)
        try:
            task = self._background.schedule(
                self._analyze_mock_speech_delivery_task(
                    session_id,
                    recording_id,
                    content,
                    transcript,
                    content_type=content_type,
                    owner=owner,
                    generation=generation,
                    omni_client=omni_client,
                    fallback=fallback,
                )
            )
        except BaseException:
            self._release_audio_settings_lease()
            self._mock_speech_feedback.pop(key, None)
            raise
        if task is None:
            self._release_audio_settings_lease()
            self._mock_speech_feedback[key] = ("completed", fallback, now)
        return fallback

    async def _analyze_mock_speech_delivery_task(
        self,
        session_id: str,
        recording_id: UUID,
        content: bytes,
        transcript: str,
        *,
        content_type: str,
        owner: str,
        generation: str,
        omni_client: Any,
        fallback: SpeechDeliveryFeedback,
    ) -> None:
        try:
            feedback = await self._analyze_mock_speech_delivery_unleased(
                session_id,
                content,
                transcript,
                content_type=content_type,
                omni_client=omni_client,
            )
            # The counter normally prevents an official settings transaction
            # from changing this token.  The explicit check also fail-closes if
            # an injected adapter or recovery path rotates it out of band.
            if generation != self.audio_settings_etag:
                feedback = fallback
            key = (owner, session_id, recording_id)
            self._mock_speech_feedback[key] = (
                "completed",
                feedback,
                datetime.now(timezone.utc).timestamp(),
            )
            runtime = self._runtimes.get((owner, session_id))
            if runtime is None:
                return
            filename_prefix = f"mock-{session_id}-{recording_id}."
            async with self._lock_for(session_id):
                records = [
                    *runtime.state.mock_session.responses,
                    *runtime.state.mock_session.attempt_history,
                ]
                record = next(
                    (item for item in records if item.audio_file.startswith(filename_prefix)),
                    None,
                )
                if record is not None:
                    record.speech_delivery = feedback
                    await self._persist(session_id, runtime.state)
                else:
                    draft = runtime.state.mock_session.answer_draft
                    if draft is not None and draft.audio_file.startswith(filename_prefix):
                        draft.speech_delivery = feedback
                        draft.updated_at = datetime.now(timezone.utc)
                        await self._persist(session_id, runtime.state)
        finally:
            self._release_audio_settings_lease()

    async def get_mock_speech_delivery_status(
        self, session_id: str, recording_id: UUID
    ) -> tuple[str, SpeechDeliveryFeedback]:
        await self._get_runtime(session_id)
        value = self._mock_speech_feedback.get((current_owner(), session_id, recording_id))
        if value is None:
            raise MockInterviewStateError("Speech feedback was not found")
        return value[0], value[1]

    @staticmethod
    def _speech_feedback_is_compliant(result: dict[str, Any]) -> bool:
        """Reject model text that crosses the coaching-only policy boundary."""
        rendered = json.dumps(result, ensure_ascii=False).casefold()
        for term in SPEECH_FEEDBACK_PROHIBITED_TERMS:
            normalized = term.casefold()
            if normalized.isascii():
                if re.search(rf"\b{re.escape(normalized)}\b", rendered):
                    return False
            elif normalized in rendered:
                return False
        return True

    @staticmethod
    def _wav_duration_seconds(content: bytes) -> float:
        try:
            with wave.open(io.BytesIO(content), "rb") as audio:
                rate = audio.getframerate()
                return audio.getnframes() / rate if rate else 0.0
        except (wave.Error, EOFError):
            return 0.0

    async def synthesize_mock_question(
        self,
        session_id: str,
        question_id: UUID,
        *,
        response_id: UUID | None = None,
    ) -> tuple[bytes, str]:
        """Synthesize a question that belongs to this owned mock session."""
        runtime = await self._get_runtime(session_id)
        question = self.current_mock_question(runtime.state)
        if question is None or question.id != question_id:
            raise MockInterviewStateError("Question is not the current mock question")
        # Before a follow-up is answered it exists only in session state, not yet
        # as a response record. Prefer that server-owned pending text so automatic
        # narration speaks what the UI is actually asking.
        speech_text = runtime.state.mock_session.pending_follow_up or question.question
        if response_id is not None:
            response = next(
                (
                    item
                    for item in runtime.state.mock_session.responses
                    if item.id == response_id and item.question_id == question_id
                ),
                None,
            )
            if response is None:
                raise MockInterviewStateError("Response is not part of the current question")
            speech_text = response.question
        if self.tts_client is None:
            raise WorkflowExecutionError("TTS client is not configured")
        started = perf_counter()
        try:
            audio, content_type = await self.tts_client.synthesize(speech_text)
        except TTSError as exc:
            self._record_debug(
                "mock_question_tts_failed",
                session_id,
                level=DebugLevel.ERROR,
                detail="tts_provider_error",
                duration_ms=(perf_counter() - started) * 1000,
            )
            raise WorkflowExecutionError("Question speech generation failed") from exc
        self._record_debug(
            "mock_question_tts_completed",
            session_id,
            detail=f"question_id={question_id}; bytes={len(audio)}",
            duration_ms=(perf_counter() - started) * 1000,
        )
        return audio, content_type

    @staticmethod
    def _live_audio_context(state: InterviewState) -> str:
        """Focused, priority-ordered context for the omni audio-direct model.

        The direct path exists to be fast, so it carries only what the model
        needs to turn a candidate utterance into a good follow-up: job/covered
        competencies (anchor), the recent stable conversation, and the most
        recent live evidence. Lower-value "advancement" info (blueprint backlog,
        historical summary, coverage guidance) is intentionally excluded to keep
        prefill small. If the focused blocks still exceed the limit, blocks are
        dropped lowest-priority first rather than truncating the tail (which
        would lose the anchor).
        """
        competencies = "、".join(state.job.competencies) or state.job.title or "目标岗位"
        covered = "、".join(
            dict.fromkeys(item.competency for item in state.evidence if item.competency.strip())
        )
        anchor = f"岗位能力：{competencies}"
        if covered:
            anchor += f"；已覆盖能力：{covered}"
        stable_segments = [
            item
            for item in state.live_interview.segments
            if item.stable and item.confirmed and item.speaker != TranscriptSpeaker.UNKNOWN
        ][-LIVE_RECENT_SEGMENT_WINDOW:]
        latest_candidate = next(
            (
                item
                for item in reversed(stable_segments)
                if item.speaker == TranscriptSpeaker.CANDIDATE
            ),
            None,
        )
        optional_parts: list[str] = []
        history_segments = [
            item
            for item in stable_segments
            if latest_candidate is None or item.id != latest_candidate.id
        ]
        if history_segments:
            speaker_labels = {
                TranscriptSpeaker.INTERVIEWER: "面试官",
                TranscriptSpeaker.CANDIDATE: "候选人",
                TranscriptSpeaker.UNKNOWN: "待确认",
            }
            optional_parts.append(
                "较早对话（只用于消歧，不得作为本轮追问主题）："
                + "；".join(
                    f"{speaker_labels.get(item.speaker, item.speaker.value)}：{item.text}"
                    for item in history_segments
                )
            )

        live_evidence = [
            f"{item.competency}：{item.signal}"
            for item in state.evidence
            if item.source == EvidenceSource.LIVE_INTERVIEW
        ][-5:]
        if live_evidence:
            optional_parts.append("已确认证据：" + "；".join(live_evidence))

        latest_block = (
            "本轮必须只围绕下方最新候选人回答提出直接追问；不得改问较早回答：\n"
            f"最新候选人回答：{latest_candidate.text}"
            if latest_candidate is not None
            else "本轮尚无已确认的候选人回答。"
        )
        parts = [anchor, *optional_parts, latest_block]

        # Drop lowest-priority optional blocks first. The competency anchor and
        # explicit latest-answer block are never displaced by older conversation.
        while len("\n".join(parts)) > OMNI_CONTEXT_CHAR_LIMIT and optional_parts:
            optional_parts.pop()
            parts = [anchor, *optional_parts, latest_block]
        if len("\n".join(parts)) > OMNI_CONTEXT_CHAR_LIMIT:
            available = max(200, OMNI_CONTEXT_CHAR_LIMIT - len(anchor) - 100)
            if len(latest_block) > available:
                head = max(80, int(available * 0.6))
                tail = max(80, available - head - 2)
                latest_block = f"{latest_block[:head]}…{latest_block[-tail:]}"
            parts = [anchor, latest_block]
        return "\n".join(parts)

    @staticmethod
    def _build_audio_direct_suggestion(
        runtime: AgentRuntime, suggestion_text: str
    ) -> QuestionSuggestion:
        """Build a reviewable suggestion without mutating session state."""

        clean = suggestion_text.strip()
        if not clean or clean.startswith("[LLM Error"):
            clean = "请再补充说明一下你刚才提到的方案权衡与结果。"
        competency = (
            runtime.state.job.competencies[0] if runtime.state.job.competencies else "综合能力"
        )
        return QuestionSuggestion(
            suggested_question=clean[:4000],
            question_type=QuestionSuggestionType.FOLLOW_UP,
            competency=competency,
            rationale="基于候选人实时语音回答生成的追问建议",
            evidence_gap="需要进一步验证回答中的行动、权衡与量化结果",
            expected_signals=["具体行动", "技术权衡", "可量化结果"],
            confidence=0.7,
        )

    async def _inject_audio_direct_suggestion(
        self,
        session_id: str,
        runtime: AgentRuntime,
        suggestion_text: str,
    ) -> None:
        """Create a QuestionSuggestion from an audio-direct model answer."""

        suggestion = self._build_audio_direct_suggestion(runtime, suggestion_text)
        async with self._lock_for(session_id):
            previous_state = runtime.state
            candidate_state = runtime.state.model_copy(deep=True)
            if candidate_state.live_interview.status not in {
                LiveInterviewStatus.ACTIVE,
                LiveInterviewStatus.PAUSED,
            }:
                # A request may finish after the interviewer ends the session.
                # Paused is accepted because the audio was captured before the
                # pause and is being drained from the durable client queue.
                return
            candidate_state.live_interview.suggestions.append(suggestion)
            self._refresh_live_question_usage(candidate_state)
            self._refresh_live_action_card(candidate_state)
            candidate_state.next_action = "Interviewer reviews the audio-direct suggestion"
            await self._persist_media_candidate(
                session_id, runtime, candidate_state, previous_state
            )

    async def _commit_live_audio_capture(
        self,
        session_id: str,
        runtime: AgentRuntime,
        *,
        segments: list[tuple[TranscriptSpeaker, str, str]] | None = None,
        suggestion: QuestionSuggestion | None = None,
        receipt: LiveAudioCaptureReceipt | None = None,
    ) -> InterviewState:
        """Atomically commit one finalized audio result and its receipt."""

        clean_segments = [
            (speaker, text.strip(), source)
            for speaker, text, source in (segments or [])
            if text.strip()
        ]
        if segments and not clean_segments:
            raise LiveInterviewStateError("Audio transcription is empty")

        async with self._lock_for(session_id):
            previous_state = runtime.state
            candidate_state = runtime.state.model_copy(deep=True)
            live = candidate_state.live_interview
            if receipt is not None:
                existing = next(
                    (item for item in live.audio_capture_receipts if item.id == receipt.id),
                    None,
                )
                if existing is not None:
                    if existing.payload_sha256 != receipt.payload_sha256:
                        raise LiveInterviewStateError(
                            "语音片段标识已用于不同内容，已拒绝重复处理"
                        )
                    return runtime.state
                if receipt.id in live.cancelled_audio_capture_ids:
                    raise LiveInterviewStateError("Audio capture was already cancelled")
                if receipt.id in live.audio_capture_guard_ids:
                    raise LiveInterviewStateError("Audio capture guard cannot contain audio")

            registered_capture = bool(
                receipt is not None and receipt.id in live.pending_audio_capture_ids
            )
            valid_new_capture = (
                live.status == LiveInterviewStatus.ACTIVE and receipt is None
            ) or (
                registered_capture
                and live.status
                in {
                    LiveInterviewStatus.ACTIVE,
                    LiveInterviewStatus.PAUSED,
                    LiveInterviewStatus.COMPLETED,
                }
            )
            if not valid_new_capture:
                raise LiveInterviewStateError(
                    "Live interview ended before the audio result could be saved"
                )

            added_segment = False
            for speaker, text, source in clean_segments:
                duplicate = self._append_live_transcript_locked(
                    candidate_state,
                    text=text,
                    speaker=speaker,
                    source=source,
                    allow_completed_capture=(
                        live.status == LiveInterviewStatus.COMPLETED
                        and registered_capture
                    ),
                )
                added_segment = added_segment or duplicate is None
            if added_segment:
                self._refresh_live_answer_boundaries(candidate_state)
                self._refresh_live_rolling_summary(candidate_state)
                self._refresh_live_coverage_guidance(candidate_state)

            if suggestion is not None and live.status != LiveInterviewStatus.COMPLETED:
                live.suggestions.append(suggestion)
                self._refresh_live_question_usage(candidate_state)
                self._refresh_live_action_card(candidate_state)
                candidate_state.next_action = (
                    "Interviewer reviews the audio-direct suggestion"
                )

            if receipt is not None:
                live.audio_capture_receipts.append(receipt)
                if receipt.id in live.pending_audio_capture_ids:
                    live.pending_audio_capture_ids.remove(receipt.id)
                live.pending_audio_capture_registered_at.pop(str(receipt.id), None)
                live.pending_audio_capture_settings_etags.pop(str(receipt.id), None)
                live.pending_audio_capture_epochs.pop(str(receipt.id), None)
                if len(live.audio_capture_receipts) > LIVE_AUDIO_RECEIPT_MAX:
                    del live.audio_capture_receipts[:-LIVE_AUDIO_RECEIPT_MAX]
            await self._persist_media_candidate(
                session_id, runtime, candidate_state, previous_state
            )
        return runtime.state

    @staticmethod
    def _merge_diarized_segments(
        segments: list[dict[str, str]],
    ) -> list[dict[str, str]]:
        """Merge consecutive utterances from the same speaker into one segment.

        The diarize model returns one entry per sentence; a long candidate
        answer spanning several sentences should become one transcript segment
        so it can be confirmed as a single piece of evidence.
        """
        merged: list[dict[str, str]] = []
        for segment in segments:
            speaker = segment.get("speaker", "unknown")
            text = segment.get("text", "").strip()
            if not text:
                continue
            if merged and merged[-1]["speaker"] == speaker:
                merged[-1]["text"] = f"{merged[-1]['text']}\n{text}"
            else:
                merged.append({"speaker": speaker, "text": text})
        return merged

    async def _transcribe_dialogue(
        self,
        session_id: str,
        runtime: AgentRuntime,
        content: bytes,
        *,
        content_type: str,
        started: float,
        receipt: LiveAudioCaptureReceipt | None = None,
    ) -> tuple[InterviewState, str]:
        """Transcribe a dialog, auto-splitting speakers, then append segments."""
        diarized = await self.omni_client.transcribe_diarize(content, content_type=content_type)
        merged = self._merge_diarized_segments(diarized)
        if not merged:
            raise LiveInterviewStateError("音频直连未能识别对话中的说话人，请重试或手动输入")
        summary_parts: list[str] = []
        transcript_segments: list[tuple[TranscriptSpeaker, str, str]] = []
        for segment in merged:
            speaker = (
                TranscriptSpeaker.CANDIDATE
                if segment["speaker"] == "candidate"
                else TranscriptSpeaker.INTERVIEWER
                if segment["speaker"] == "interviewer"
                else TranscriptSpeaker.UNKNOWN
            )
            text = segment["text"]
            summary_parts.append(text)
            transcript_segments.append((speaker, text, "audio_direct"))
        state = await self._commit_live_audio_capture(
            session_id,
            runtime,
            segments=transcript_segments,
            receipt=receipt,
        )
        self._record_debug(
            "audio_direct_dialogue",
            session_id,
            detail=f"segments={len(merged)}; speakers={len({s['speaker'] for s in merged})}",
            duration_ms=(perf_counter() - started) * 1000,
        )
        summary = " | ".join(summary_parts)
        return state, summary

    async def save_live_audio(
        self,
        session_id: str,
        content: bytes,
        *,
        extension: str = "wav",
        recording_id: UUID | None = None,
        expected_audio_revision: int,
    ) -> InterviewState:
        """Persist an immutable whole-session recording part.

        A recording can stop at pause, session switch, or app shutdown and then
        resume later. Parts use a client-generated idempotency ID so a response
        lost after a successful upload cannot create a duplicate on retry.
        ``audio_file`` remains the latest part for backward-compatible clients.
        Ownership is enforced by ``_get_runtime`` (owner-scoped), so a foreign
        session raises 404.
        """
        runtime = await self._get_runtime(session_id)
        async with self._lock_for(session_id):
            previous_state = runtime.state
            candidate_state = runtime.state.model_copy(deep=True)
            live = candidate_state.live_interview
            payload_sha256 = hashlib.sha256(content).hexdigest()
            if expected_audio_revision != live.audio_archive_revision:
                raise LiveInterviewStateError(
                    "原始录音集合已在另一窗口变更；旧录音分片未写入"
                )
            if not live.consent_confirmed:
                raise LiveInterviewStateError("未确认知情同意前无法保存全场录音")
            ext = (extension or "wav").lstrip(".").lower()
            if ext not in {"wav", "webm", "m4a"}:
                ext = "wav"

            # Preserve the original endpoint contract for older browser/API
            # clients that do not send a part ID. Once a segmented manifest
            # exists we still append rather than destroy its immutable parts.
            if recording_id is None and not live.audio_parts:
                filename = f"{session_id}.{ext}"
                target = self._recordings_dir / filename
                temp = self._recordings_dir / f"{filename}.tmp"
                self._ensure_private_recordings_dir()
                backup: Path | None = None
                if target.exists():
                    backup = target.with_name(
                        f".{target.name}.{uuid4().hex}.replace-pending"
                    )
                    os.replace(target, backup)
                try:
                    self._write_then_replace(temp, target, content)
                except OSError:
                    temp.unlink(missing_ok=True)
                    if backup is not None and backup.exists():
                        os.replace(backup, target)
                    raise
                saved_at = datetime.now(timezone.utc)
                live.audio_file = filename
                live.audio_size_bytes = len(content)
                live.audio_payload_sha256 = payload_sha256
                live.audio_saved_at = saved_at
                try:
                    await self._persist_media_candidate(
                        session_id, runtime, candidate_state, previous_state
                    )
                except BaseException:
                    recovery_pending = (
                        current_owner(), session_id
                    ) in self._runtime_recovery_required
                    durable = runtime.state.live_interview
                    committed = (
                        not recovery_pending
                        and durable.audio_file == filename
                        and durable.audio_payload_sha256 == payload_sha256
                    )
                    if committed:
                        if backup is not None:
                            try:
                                backup.unlink(missing_ok=True)
                            except OSError:
                                logger.warning(
                                    "Could not remove committed recording backup %s",
                                    backup.name,
                                )
                        return runtime.state
                    raise
                if backup is not None:
                    try:
                        backup.unlink(missing_ok=True)
                    except OSError:
                        logger.warning(
                            "Could not remove superseded recording backup %s",
                            backup.name,
                        )
                for old_ext in {"wav", "webm", "m4a"} - {ext}:
                    stale = self._recordings_dir / f"{session_id}.{old_ext}"
                    try:
                        stale.unlink(missing_ok=True)
                    except OSError as exc:
                        logger.warning(
                            "Could not remove superseded legacy recording %s: %s",
                            stale.name,
                            type(exc).__name__,
                        )
                self._record_debug(
                    "live_audio_saved",
                    session_id,
                    detail=f"bytes={len(content)}; file={live.audio_file}; legacy=true",
                )
                return runtime.state

            part_id = recording_id or uuid4()
            existing = next(
                (part for part in live.audio_parts if part.id == part_id), None
            )
            if existing is not None:
                if existing.audio_size_bytes != len(content) or (
                    existing.payload_sha256
                    and existing.payload_sha256 != payload_sha256
                ):
                    raise LiveInterviewStateError("录音分片标识冲突，已拒绝覆盖原始音频")
                existing_path = self.get_live_audio_path(session_id, existing.audio_file)
                if existing_path is not None:
                    stored_sha256 = self._file_sha256(existing_path)
                    if stored_sha256 != payload_sha256:
                        raise LiveInterviewStateError(
                            "录音分片标识冲突，已拒绝覆盖原始音频"
                        )
                else:
                    existing_path = self._recordings_dir / existing.audio_file
                    temp = self._recordings_dir / f"{existing.audio_file}.tmp"
                    self._ensure_private_recordings_dir()
                    self._write_then_replace(temp, existing_path, content)
                existing.payload_sha256 = payload_sha256
                if live.audio_file == existing.audio_file:
                    live.audio_payload_sha256 = payload_sha256
                await self._persist_media_candidate(
                    session_id, runtime, candidate_state, previous_state
                )
                return runtime.state

            # Migrate a pre-segmentation recording into the manifest before
            # appending a new part. Its old filename remains valid and local.
            if live.audio_file and not live.audio_parts:
                legacy_path = self.get_live_audio_path(session_id, live.audio_file)
                if legacy_path is not None:
                    live.audio_parts.append(
                        LiveAudioPart(
                            audio_file=live.audio_file,
                            audio_size_bytes=live.audio_size_bytes or legacy_path.stat().st_size,
                            payload_sha256=(
                                live.audio_payload_sha256
                                or self._file_sha256(legacy_path)
                            ),
                            audio_saved_at=live.audio_saved_at
                            or datetime.now(timezone.utc),
                        )
                    )

            self._ensure_private_recordings_dir()
            filename = f"{session_id}-{part_id}.{ext}"
            target = self._recordings_dir / filename
            temp = self._recordings_dir / f"{filename}.tmp"
            created_target = not target.exists()
            if target.exists():
                if self._file_sha256(target) != payload_sha256:
                    raise LiveInterviewStateError(
                        "录音分片标识冲突，已拒绝覆盖原始音频"
                    )
            else:
                self._write_then_replace(temp, target, content)
            saved_at = datetime.now(timezone.utc)
            live.audio_parts.append(
                LiveAudioPart(
                    id=part_id,
                    audio_file=filename,
                    audio_size_bytes=len(content),
                    payload_sha256=payload_sha256,
                    audio_saved_at=saved_at,
                )
            )
            live.audio_file = filename
            live.audio_size_bytes = len(content)
            live.audio_payload_sha256 = payload_sha256
            live.audio_saved_at = saved_at
            try:
                await self._persist_media_candidate(
                    session_id, runtime, candidate_state, previous_state
                )
            except BaseException:
                recovery_pending = (
                    current_owner(), session_id
                ) in self._runtime_recovery_required
                durable_part = next(
                    (
                        part
                        for part in runtime.state.live_interview.audio_parts
                        if part.id == part_id
                    ),
                    None,
                )
                if (
                    not recovery_pending
                    and durable_part is not None
                    and durable_part.payload_sha256 == payload_sha256
                ):
                    return runtime.state
                if not recovery_pending and created_target:
                    # _persist_media_candidate already reconciled this file
                    # against the durable manifest. Keep this idempotent guard
                    # for stores whose recovery hook is customized.
                    try:
                        target.unlink(missing_ok=True)
                    except OSError:
                        self._runtime_recovery_required.add(
                            (current_owner(), session_id)
                        )
                raise
        self._record_debug(
            "live_audio_saved",
            session_id,
            detail=f"bytes={len(content)}; file={live.audio_file}",
        )
        return runtime.state

    def get_live_audio_path(self, session_id: str, stored_filename: str = "") -> Path | None:
        """Return the on-disk path of a session's recording, if present.

        The caller (owner-scoped route) already validated ownership via
        ``get_state``. Only the filename persisted in that owner-scoped state is
        accepted; scanning by session id could serve a superseded recording.
        """
        name = Path(stored_filename).name
        if name != stored_filename:
            return None
        legacy_names = {f"{session_id}.{ext}" for ext in ("wav", "webm", "m4a")}
        segmented_name = re.fullmatch(
            rf"{re.escape(session_id)}-[0-9a-f-]{{36}}\.(wav|webm|m4a)", name
        )
        if name not in legacy_names and segmented_name is None:
            return None
        candidate = self._recordings_dir / name
        return candidate if candidate.is_file() else None

    async def open_live_audio_file(
        self, session_id: str, *, recording_id: UUID | None = None
    ) -> tuple[BinaryIO, str] | None:
        """Open the authoritative recording while holding the session lock.

        Returning an already-open descriptor prevents a concurrent replace or
        delete from making a path-based response serve uncommitted bytes (or a
        transient 404). On POSIX the descriptor survives rename; on Windows the
        mutation safely waits/fails until the download releases it.
        """

        runtime = await self._get_runtime(session_id)
        async with self._lock_for(session_id):
            live = runtime.state.live_interview
            if recording_id is None:
                filename = live.audio_file
                expected_hash = live.audio_payload_sha256
            else:
                part = next(
                    (item for item in live.audio_parts if item.id == recording_id),
                    None,
                )
                if part is None:
                    return None
                filename = part.audio_file
                expected_hash = part.payload_sha256
            path = self.get_live_audio_path(session_id, filename)
            if path is None:
                return None
            if expected_hash and self._file_sha256(path) != expected_hash:
                raise LiveInterviewStateError(
                    "录音文件与持久化清单不一致；请先完成存储恢复"
                )
            try:
                return path.open("rb"), path.name
            except OSError:
                return None

    async def delete_live_audio(
        self,
        session_id: str,
        *,
        expected_audio_revision: int,
        operation_id: UUID,
    ) -> InterviewState:
        """Delete every persisted whole-session recording part for its owner."""

        runtime = await self._get_runtime(session_id)
        quarantined: list[tuple[Path, Path]] = []
        async with self._lock_for(session_id):
            previous_state = runtime.state
            candidate_state = runtime.state.model_copy(deep=True)
            live = candidate_state.live_interview
            if live.last_audio_delete_operation_id == operation_id:
                # The original response may have been lost. Reconcile any
                # crash residue, but never delete parts appended afterwards.
                self._reconcile_live_audio_for_state(
                    session_id, runtime.state, strict=True
                )
                return runtime.state
            if expected_audio_revision != live.audio_archive_revision:
                raise LiveInterviewStateError(
                    "原始录音集合已在另一窗口变更；本窗口未执行删除"
                )
            # Fence uploads that were created before this explicit deletion.
            # The revision is persisted in the same critical section as the
            # manifest update, so a delayed request cannot resurrect a part.
            live.audio_archive_revision += 1
            live.last_audio_delete_operation_id = operation_id
            filenames = {part.audio_file for part in live.audio_parts}
            if live.audio_file:
                filenames.add(live.audio_file)
            try:
                for filename in filenames:
                    path = self.get_live_audio_path(session_id, filename)
                    if path is None:
                        continue
                    quarantine = path.with_name(
                        f".{path.name}.{uuid4().hex}.delete-pending"
                    )
                    os.replace(path, quarantine)
                    quarantined.append((path, quarantine))
            except OSError as exc:
                try:
                    self._reconcile_live_audio_for_state(
                        session_id, runtime.state, strict=True
                    )
                except BaseException:  # noqa: BLE001 - fence until recovery
                    self._runtime_recovery_required.add(
                        (current_owner(), session_id)
                    )
                raise LiveInterviewStateError(
                    "录音文件暂时无法安全删除，原清单保持不变"
                ) from exc

            live.audio_parts = []
            live.audio_file = ""
            live.audio_size_bytes = 0
            live.audio_payload_sha256 = ""
            live.audio_saved_at = None
            try:
                await self._persist(session_id, candidate_state)
            except BaseException:
                await self._recover_live_state_after_persist_error(
                    session_id, runtime, previous_state
                )
                recovery_pending = (
                    current_owner(), session_id
                ) in self._runtime_recovery_required
                if not recovery_pending:
                    try:
                        self._reconcile_live_audio_for_state(
                            session_id, runtime.state, strict=True
                        )
                    except BaseException:  # noqa: BLE001 - retry on next lock entry
                        self._runtime_recovery_required.add(
                            (current_owner(), session_id)
                        )
                        recovery_pending = True
                durable = runtime.state.live_interview
                committed = (
                    not recovery_pending
                    and durable.audio_archive_revision == live.audio_archive_revision
                    and not durable.audio_parts
                    and not durable.audio_file
                )
                if not committed and not recovery_pending:
                    raise
                if recovery_pending:
                    # Keep files hidden and non-destructive until the next
                    # session-lock entry reloads SQLite and reconciles them.
                    raise
            else:
                runtime.state = candidate_state
            # Return success only once every transaction artifact for this
            # session is gone. This also removes an older replacement backup,
            # so DELETE cannot leave superseded voice data on disk.
            self._reconcile_live_audio_for_state(
                session_id, runtime.state, strict=True
            )
        self._record_debug("live_audio_deleted", session_id, detail="all_parts=true")
        return runtime.state
