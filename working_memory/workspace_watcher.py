"""
SOFi WorkspaceWatcher — Background Proactive Notification Monitor

Runs as a daemon thread. Every `poll_interval_s` seconds (configured in
config.workspace_watcher_poll_interval_s, default 5s) it checks the
AgenticWorkspace for:

  1. Items with notify=True and status != HANDLED
  2. Reminders / alarms whose due_at is within the next poll window

When a fire-eligible item is found, it calls the injected `proactive_callback`
with the item, then marks the item as HANDLED so it is not fired again.

Priority Levels
===============
  URGENT — fire immediately, regardless of conversation state
  NORMAL — fire only if the user has been inactive for gap_threshold_s seconds
  LOW    — do NOT fire proactively; these surface naturally via the Working
            Context snapshot (the mandatory injection slice always includes
            pending LOW items so SOFi sees them at the next user turn)

The callback signature is:
    proactive_callback(item: WorkspaceItem) -> None

The caller (MemoryManager or the main assistant loop) injects this callback
and is responsible for actually activating SOFi (e.g., triggering an LLM call
with the current Working Context as the system prompt).
"""

import threading
import time
from datetime import datetime
from typing import Callable, Optional

from memory.config import config
from memory.working_memory.working_context import (
    WorkingContextManager,
    WorkspaceItem,
    WorkspaceItemStatus,
    NotifyPriority,
    WorkspaceItemType,
)
from utils.logger import UniversalLogger

logger = UniversalLogger.get_logger("workspace_watcher")


class WorkspaceWatcher:
    """
    Background daemon thread — monitors the Agentic Workspace for items
    that require proactive user notification.

    Usage (from MemoryManager.setup()):
        watcher = WorkspaceWatcher(
            context_manager=self.context_manager,
            proactive_callback=self._handle_proactive,
        )
        watcher.start()

    Shutdown (from MemoryManager.shutdown()):
        watcher.stop()
    """

    def __init__(
        self,
        context_manager: WorkingContextManager,
        proactive_callback: Callable[[WorkspaceItem], None],
    ) -> None:
        """
        Args:
            context_manager:    The live WorkingContextManager.
            proactive_callback: Called with the WorkspaceItem when a
                                notification should fire. The callback must
                                be thread-safe (it is called from the watcher
                                daemon thread, NOT from the main thread).
        """
        self._ctx               = context_manager
        self._callback          = proactive_callback

        # All timing from config — no magic numbers in code
        self._poll_interval_s: int = config.workspace_watcher_poll_interval_s
        self._gap_threshold_s:  int = config.workspace_watcher_gap_threshold_s

        self._running: bool         = False
        self._thread: Optional[threading.Thread] = None

        # Track time of last user message for NORMAL-priority gap detection
        self._last_user_activity: datetime = datetime.now()

    # ─────────────────────────────────────────────────────────────────────────
    # Lifecycle
    # ─────────────────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the background watcher daemon thread."""
        if self._running:
            logger.warning("[watcher] Already running — ignoring start()")
            return

        self._running = True
        self._thread  = threading.Thread(
            target=self._watch_loop,
            name="workspace-watcher",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            f"[watcher] Started "
            f"(poll={self._poll_interval_s}s gap={self._gap_threshold_s}s)"
        )

    def stop(self) -> None:
        """Signal the watcher thread to stop. Does not block."""
        self._running = False
        logger.info("[watcher] Stop signal sent")

    def record_user_activity(self) -> None:
        """
        Call this every time the user sends a message.
        Used to determine whether the conversation is in an 'active' state
        (within gap_threshold_s) or an idle state where NORMAL notifications
        can fire.
        """
        self._last_user_activity = datetime.now()

    # ─────────────────────────────────────────────────────────────────────────
    # Main loop
    # ─────────────────────────────────────────────────────────────────────────

    def _watch_loop(self) -> None:
        logger.debug("[watcher] Loop started")
        while self._running:
            try:
                self._check_pending_notifications()
                self._check_due_reminders()
            except Exception as exc:
                # Never let the watcher crash — log and continue
                logger.error(f"[watcher] Unexpected error: {exc}", exc_info=True)

            time.sleep(self._poll_interval_s)

        logger.debug("[watcher] Loop exited")

    # ─────────────────────────────────────────────────────────────────────────
    # Checks
    # ─────────────────────────────────────────────────────────────────────────

    def _check_pending_notifications(self) -> None:
        """Check all workspace items with notify=True."""
        pending = self._ctx.workspace.get_pending_notifications()
        for item in pending:
            if self._should_fire(item):
                self._fire(item)

    def _check_due_reminders(self) -> None:
        """Check reminders / alarms that are due within the next poll window."""
        due = self._ctx.workspace.get_due_reminders(
            within_seconds=self._poll_interval_s * 2  # small lookahead
        )
        for item in due:
            # Due reminders always fire (they represent user-set time commitments)
            if item.status not in (
                WorkspaceItemStatus.HANDLED,
                WorkspaceItemStatus.COMPLETED,
            ):
                logger.info(f"[watcher] Due reminder: '{item.title}'")
                self._fire(item)

    # ─────────────────────────────────────────────────────────────────────────
    # Fire decision
    # ─────────────────────────────────────────────────────────────────────────

    def _should_fire(self, item: WorkspaceItem) -> bool:
        """
        Decide whether this item should trigger a proactive notification now.

        URGENT → always fire immediately
        NORMAL → fire only if user has been inactive for gap_threshold_s seconds
        LOW    → never fire proactively (surfaced via mandatory injection)
        """
        if item.notify_priority == NotifyPriority.URGENT:
            return True

        if item.notify_priority == NotifyPriority.NORMAL:
            seconds_idle = (datetime.now() - self._last_user_activity).total_seconds()
            return seconds_idle >= self._gap_threshold_s

        # LOW priority — surface at next user turn, not via watcher
        return False

    # ─────────────────────────────────────────────────────────────────────────
    # Fire
    # ─────────────────────────────────────────────────────────────────────────

    def _fire(self, item: WorkspaceItem) -> None:
        """
        Mark the item as HANDLED and invoke the proactive callback.
        The callback is responsible for activating SOFi.
        """
        logger.info(
            f"[watcher] Firing notification: '{item.title}' "
            f"(type={item.type.value} priority={item.notify_priority.value})"
        )
        # Mark HANDLED first so a second poll cycle doesn't double-fire
        self._ctx.workspace.update_item(
            item.id,
            status=WorkspaceItemStatus.HANDLED,
            notify=False,
        )

        try:
            self._callback(item)
        except Exception as exc:
            logger.error(
                f"[watcher] proactive_callback raised for '{item.title}': {exc}",
                exc_info=True,
            )
