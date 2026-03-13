"""
webapp/server.py — FastAPI dashboard server.

Routes:
    GET  /                  → index.html
    GET  /api/state         → sensors + pipeline + emotion (cold path, ~1 Hz poll)
    GET  /api/sim-history   → last 300 similarity chunks for chart pre-fill
    WS   /ws/audio          → hot path: similarity score, FSM state, peak amplitude
"""

import asyncio
import json
import time
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

import serial_reader
import state_bus

_STATIC = Path(__file__).parent / "static"

app = FastAPI(title="CHAOS Dashboard")
app.mount("/static", StaticFiles(directory=str(_STATIC)), name="static")


@app.get("/", response_class=HTMLResponse)
async def index():
    return (_STATIC / "index.html").read_text()


@app.get("/api/state")
async def api_state():
    return JSONResponse({
        "sensors":  serial_reader.robot_state,
        "pipeline": state_bus.pipeline_state,
        "emotion":  state_bus.get_current_emotion(),
        "ts":       time.time(),
    })


@app.post("/api/trigger")
async def api_trigger():
    state_bus.manual_trigger.set()
    return JSONResponse({"ok": True})


@app.get("/api/sim-history")
async def api_sim_history():
    return JSONResponse({"history": list(state_bus.sim_history)})


@app.websocket("/ws/audio")
async def ws_audio(websocket: WebSocket):
    await websocket.accept()

    async def send_chunk(payload: dict):
        await websocket.send_text(json.dumps({**payload, "ts": time.time()}))

    state_bus.subscribe_audio(send_chunk)
    try:
        while True:
            await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
    except (WebSocketDisconnect, asyncio.TimeoutError, Exception):
        pass
    finally:
        state_bus.unsubscribe_audio(send_chunk)
