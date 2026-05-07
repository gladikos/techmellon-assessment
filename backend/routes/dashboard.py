"""
Dashboard routes:
- GET /                serves the HTML observation UI
- WebSocket /ws/run    streams refinement loop events live to the browser

The orchestrator is sync-blocking. We run it in a daemon thread; events
flow through a thread-safe queue, which the async WebSocket consumer
pulls from and forwards as JSON.
"""

from __future__ import annotations

import asyncio
import queue
import threading
from pathlib import Path

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from refinement.orchestrator import run_all_scenarios, run_pipeline

router = APIRouter()

DASHBOARD_HTML = Path(__file__).resolve().parent.parent / "dashboard.html"


@router.get("/", include_in_schema=False)
async def dashboard_page():
    """Serve the static dashboard HTML."""
    return FileResponse(DASHBOARD_HTML, media_type="text/html")


@router.websocket("/ws/run")
async def ws_run(ws: WebSocket):
    """
    Bidirectional WebSocket for running the refinement loop.

    Protocol:
      Client → Server (one message to start):
        {"action": "start_one", "scenario_id": "<id>"}
        {"action": "start_all"}

      Server → Client (many messages, until done):
        {"type": "<event_type>", "payload": {...}}

      The server closes the connection after the run finishes.
    """
    await ws.accept()
    from backend.main import subscribe_to_logs, unsubscribe_from_logs
    log_q = subscribe_to_logs()
    try:
        msg = await ws.receive_json()
        action = msg.get("action")

        # Thread-safe queue for events emitted by the (sync) orchestrator.
        event_q: queue.Queue = queue.Queue()
        _DONE = object()  # sentinel to signal the worker has finished

        def callback(event_type: str, payload: dict) -> None:
            event_q.put({"type": event_type, "payload": payload})

        def worker() -> None:
            try:
                if action == "start_one":
                    scenario_id = msg.get("scenario_id")
                    if not scenario_id:
                        event_q.put({
                            "type": "error",
                            "payload": {"message": "start_one requires scenario_id"},
                        })
                        return
                    run_pipeline(scenario_id, event_callback=callback)
                elif action == "start_all":
                    run_all_scenarios(event_callback=callback)
                else:
                    event_q.put({
                        "type": "error",
                        "payload": {"message": f"unknown action: {action}"},
                    })
            except Exception as e:
                event_q.put({
                    "type": "error",
                    "payload": {"message": f"{type(e).__name__}: {e}"},
                })
            finally:
                event_q.put(_DONE)

        threading.Thread(target=worker, daemon=True).start()

        loop = asyncio.get_event_loop()
        done = asyncio.Event()

        async def consume_events():
            while True:
                event = await loop.run_in_executor(None, event_q.get)
                if event is _DONE:
                    done.set()
                    break
                await ws.send_json(event)

        async def consume_logs():
            while True:
                try:
                    item = await loop.run_in_executor(
                        None, lambda: log_q.get(timeout=0.1)
                    )
                    await ws.send_json({"type": "backend_log", "payload": item})
                except queue.Empty:
                    if done.is_set():
                        # Drain any items that arrived just before done was set.
                        while True:
                            try:
                                item = log_q.get_nowait()
                                await ws.send_json({"type": "backend_log", "payload": item})
                            except queue.Empty:
                                break
                        break

        await asyncio.gather(consume_events(), consume_logs())

    except WebSocketDisconnect:
        # Client closed the tab — let the worker keep running in the
        # background but stop trying to send. The orchestrator will
        # complete and write its logs as usual.
        return
    except Exception as e:
        try:
            await ws.send_json({
                "type": "error",
                "payload": {"message": f"{type(e).__name__}: {e}"},
            })
        except Exception:
            pass
    finally:
        unsubscribe_from_logs(log_q)
        try:
            await ws.close()
        except Exception:
            pass