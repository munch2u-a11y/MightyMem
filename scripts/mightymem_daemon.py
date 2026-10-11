#!/usr/bin/env python3
"""MightyMem 24/7 Background Daemon Runner Script.

Runs the autonomous event-driven HelixPulseLoop in the background continuously,
handling JIT context compilation, topical orientation tracking, rolling factual
summarization, and background memory maintenance.
"""
from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import time
from pathlib import Path

from context_loom import ContextLoom, OllamaEmbeddingProvider, CompilerConfig
from context_loom.helix_pulse import HelixPulseLoop
from context_loom.habitus_agent import HabitusLoomAgent, LocalModels

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("mightymem_daemon")


def parse_args():
    parser = argparse.ArgumentParser(description="MightyMem 24/7 Autonomous Background Daemon")
    parser.add_argument("--db", default="mightymem_daemon.sqlite", help="Path to persistent SQLite database")
    parser.add_argument("--scope", default="default", help="Memory scope partition")
    parser.add_argument("--model", default="granite4.1:8b", help="Local LLM model name via Ollama")
    parser.add_argument("--embed-model", default="qwen3-embedding:0.6b", help="Local embedding model name via Ollama")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434", help="Ollama API base URL")
    parser.add_argument("--active-interval", type=float, default=10.0, help="Pulse interval during active interaction (seconds)")
    parser.add_argument("--regular-interval", type=float, default=30.0, help="Pulse interval during autonomous tasks (seconds)")
    parser.add_argument("--resting-interval", type=float, default=900.0, help="Pulse interval during resting/idle phase (seconds)")
    return parser.parse_args()


def main():
    args = parse_args()
    db_path = Path(args.db).resolve()
    logger.info(f"Starting MightyMem 24/7 Background Daemon using database '{db_path}'...")

    embedder = OllamaEmbeddingProvider(model=args.embed_model, endpoint=args.ollama_url)
    loom = ContextLoom(
        db_path,
        embedding_provider=embedder,
        config=CompilerConfig(minimum_utility=0.0),
    )

    loop = HelixPulseLoop(
        loom,
        scope=args.scope,
        active_interval=args.active_interval,
        regular_interval=args.regular_interval,
        resting_interval=args.resting_interval,
    )

    # Shutdown signal handlers
    def shutdown_handler(signum, frame):
        logger.info(f"Received signal {signum}. Stopping MightyMem 24/7 daemon...")
        loop.stop()
        loom.close()
        logger.info("MightyMem daemon shut down cleanly.")
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    loop.start()
    logger.info(f"MightyMem 24/7 Daemon is active! Pulse count: {loop.pulse_count}, State: {loop.state}")

    try:
        while True:
            time.sleep(5.0)
            logger.debug(f"Daemon heartbeat: pulse={loop.pulse_count}, state={loop.state}")
    except KeyboardInterrupt:
        shutdown_handler(signal.SIGINT, None)


if __name__ == "__main__":
    main()
