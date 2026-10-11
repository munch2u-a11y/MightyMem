"""Helix Pulse Autonomous 24/7 Background Loop for MightyMem (ContextLoom & Habitus AI).

Provides an event-driven, multi-cadence autonomous consciousness loop for
local memory agents, supporting JIT context compilation, topical frame updates,
rolling fact summarization, receipt-backed actions, and post-pulse subconscious hooks.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import logging
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from .engine import ContextLoom
from .models import RecordKind, EvidenceStatus
from .topical import TopicalTracker, TopicalFrame
from .summarizer import RollingFactSummarizer

logger = logging.getLogger("context_loom.helix_pulse")


class PulseState:
    """Cadence states for Helix 24/7 pulse loop."""
    ACTIVE = "ACTIVE"       # 10s fast cadence during active conversation
    REGULAR = "REGULAR"     # 30s cadence during autonomous task work
    RESTING = "RESTING"     # 15m (900s) low-cadence awake/idle pulse
    DORMANT = "DORMANT"     # Sleep window / nightly memory consolidation


@dataclass
class PulseEvent:
    """Structured event queued into Helix pulse loop."""
    kind: str
    content: str
    scope: str = "default"
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)


class HelixPulseLoop:
    """Autonomous event-driven 24/7 consciousness loop integrating ContextLoom.

    Integrates JIT Context Packet Compilation, topical orientation updates,
    rolling fact summarization, receipt-backed tool dispatches, and
    post-pulse subconscious memory maintenance hooks.
    """

    def __init__(
        self,
        loom: ContextLoom,
        scope: str = "default",
        agent: Optional[Any] = None,
        active_interval: float = 10.0,
        regular_interval: float = 30.0,
        resting_interval: float = 900.0,
        active_timeout: float = 15.0,
        regular_timeout: float = 60.0,
        sleep_schedule: Optional[Tuple[int, int, int, int]] = None,
    ) -> None:
        self.loom = loom
        self.scope = scope
        self.agent = agent

        # Cadence intervals (seconds)
        self.active_interval = active_interval
        self.regular_interval = regular_interval
        self.resting_interval = resting_interval

        # Cadence timeouts (seconds)
        self.active_timeout = active_timeout
        self.regular_timeout = regular_timeout

        # Sleep schedule (start_hour, start_minute, end_hour, end_minute)
        self.sleep_schedule = sleep_schedule

        # Loop state
        self._state = PulseState.RESTING
        self._pulse_count = 0
        self._last_event_time: float = 0.0
        self._last_activity_time: float = 0.0

        # Subsystems
        self.topical_tracker = TopicalTracker(self.loom.store)
        self.rolling_summarizer = RollingFactSummarizer(self.loom)

        # Event queue & re-entrancy locks
        self._event_queue: List[PulseEvent] = []
        self._event_lock = threading.Lock()
        self._loom_lock = threading.Lock()

        # Post-pulse hooks
        self._post_pulse_hooks: List[Callable[[int, List[PulseEvent]], None]] = []

        # Thread lifecycle
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._wake_event = threading.Event()

    @property
    def state(self) -> str:
        return self._state

    @property
    def pulse_count(self) -> int:
        return self._pulse_count

    @property
    def pending_events_count(self) -> int:
        with self._event_lock:
            return len(self._event_queue)

    @property
    def is_running(self) -> bool:
        return self._running and (self._thread is not None and self._thread.is_alive())

    def register_post_pulse_hook(
        self, hook: Callable[[int, List[PulseEvent]], None]
    ) -> None:
        """Register a subconscious background callback executed after each pulse."""
        self._post_pulse_hooks.append(hook)

    def emit_event(
        self,
        kind: str,
        content: str,
        scope: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> PulseEvent:
        """Enqueue an event (user message, observation, signal) into the pulse loop."""
        event_scope = scope or self.scope
        event = PulseEvent(
            kind=kind,
            content=content,
            scope=event_scope,
            timestamp=time.time(),
            metadata=metadata or {},
        )

        with self._event_lock:
            self._event_queue.append(event)
            self._last_event_time = time.time()
            if self._state != PulseState.DORMANT:
                self._state = PulseState.ACTIVE

        # Log incoming user message directly into ContextLoom store
        if kind == "user_message":
            with self._loom_lock:
                self.loom.log(
                    speaker="User",
                    content=content,
                    kind=RecordKind.USER,
                    scope=event_scope,
                    evidence_status=EvidenceStatus.USER_REPORTED,
                )

        self._wake_event.set()
        return event

    def _check_state_transitions(self) -> None:
        """Evaluate cadence state transitions based on timeouts and schedule."""
        now = time.time()
        dt_now = datetime.now()

        # Check sleep schedule
        if self.sleep_schedule:
            sh, sm, eh, em = self.sleep_schedule
            curr_minutes = dt_now.hour * 60 + dt_now.minute
            start_minutes = sh * 60 + sm
            end_minutes = eh * 60 + em

            is_sleep = False
            if start_minutes > end_minutes:
                is_sleep = curr_minutes >= start_minutes or curr_minutes < end_minutes
            else:
                is_sleep = start_minutes <= curr_minutes < end_minutes

            if is_sleep and self.pending_events_count == 0:
                self._state = PulseState.DORMANT
                return

        # Cadence timeouts
        if self._state == PulseState.ACTIVE:
            if (now - self._last_event_time) > self.active_timeout:
                self._state = PulseState.REGULAR
        elif self._state == PulseState.REGULAR:
            if (now - self._last_event_time) > self.regular_timeout:
                self._state = PulseState.RESTING

    def step(self) -> Dict[str, Any]:
        """Execute a single pulse cycle synchronously."""
        with self._event_lock:
            events_to_process = list(self._event_queue)
            self._event_queue.clear()

        self._pulse_count += 1
        self._check_state_transitions()

        # Build query context from pending events or latest conversational input
        query_text = (
            " ".join(e.content for e in events_to_process)
            if events_to_process
            else "Autonomous pulse status check and memory retrieval."
        )

        # Compile JIT Context Packet via ContextLoom
        with self._loom_lock:
            packet = self.loom.compiler.compile(current_input=query_text, scope=self.scope)
            admitted = [m for m in packet.manifest if m.admitted]

        # Execute attached agent perception/step if available
        agent_response = None
        if self.agent is not None and hasattr(self.agent, "perceive_and_act"):
            try:
                agent_response = self.agent.perceive_and_act(
                    query=query_text,
                    context_header=packet.rendered,
                    events=events_to_process,
                )
            except Exception as exc:
                logger.error(f"Error executing agent step during pulse {self._pulse_count}: {exc}")

        # Periodic subconscious memory maintenance pass (reflection & rolling fact summary)
        if self._pulse_count % 10 == 0:
            try:
                with self._loom_lock:
                    self.loom.reflect(scope=self.scope)
            except Exception as exc:
                logger.error(f"Subconscious reflection failed on pulse {self._pulse_count}: {exc}")

        # Execute registered post-pulse subconscious hooks
        for hook in self._post_pulse_hooks:
            try:
                hook(self._pulse_count, events_to_process)
            except Exception as exc:
                logger.error(f"Post-pulse hook failed on pulse {self._pulse_count}: {exc}")

        self._last_activity_time = time.time()
        return {
            "pulse_count": self._pulse_count,
            "state": self._state,
            "events_processed": len(events_to_process),
            "context_header": packet.rendered,
            "topical_orientation": self.topical_tracker.frame().rendered,
            "admitted_evidence_count": len(admitted),
            "token_count": packet.token_count,
            "agent_response": agent_response,
        }

    def start(self) -> None:
        """Start the background daemon thread for continuous pulse execution."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop_worker, daemon=True, name="HelixPulseThread")
        self._thread.start()
        logger.info(f"HelixPulseLoop 24/7 background daemon started for scope '{self.scope}'.")

    def stop(self) -> None:
        """Stop the background daemon thread."""
        self._running = False
        self._wake_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        logger.info("HelixPulseLoop 24/7 background daemon stopped.")

    def _loop_worker(self) -> None:
        """Internal worker loop running in background thread."""
        while self._running:
            try:
                self.step()
            except Exception as exc:
                logger.error(f"Exception in HelixPulseLoop worker: {exc}")

            # Calculate sleep interval based on current cadence state
            if self._state == PulseState.ACTIVE:
                interval = self.active_interval
            elif self._state == PulseState.REGULAR:
                interval = self.regular_interval
            elif self._state == PulseState.DORMANT:
                interval = 60.0
            else:
                interval = self.resting_interval

            self._wake_event.wait(timeout=interval)
            self._wake_event.clear()
