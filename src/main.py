from __future__ import annotations

import logging
import signal
import sys
import time

from .orchestrator import HalloweenOrchestrator


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def main() -> int:
    configure_logging()
    try:
        orchestrator = HalloweenOrchestrator()
    except Exception as exc:  # noqa: BLE001
        logging.exception("Failed to initialize application: %s", exc)
        return 1

    def handle_exit(signum, frame):  # type: ignore[return-type]
        logging.info("Received signal %s, shutting down", signum)
        orchestrator.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_exit)
    signal.signal(signal.SIGTERM, handle_exit)

    try:
        orchestrator.start()
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logging.info("Interrupted by user")
    finally:
        orchestrator.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
