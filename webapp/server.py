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

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
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


@app.post("/api/stop-listen")
async def api_stop_listen():
    state_bus.stop_listening.set()
    return JSONResponse({"ok": True})


@app.post("/api/drive")
async def api_drive(request: Request):
    body = await request.json()
    t = max(-1.0, min(1.0, float(body.get("throttle", 0))))
    s = max(-1.0, min(1.0, float(body.get("steering", 0))))
    ti = max(-100, min(100, int(t * 100)))
    si = max(-100, min(100, int(s * 100)))
    serial_reader.send_command(f"CMD,DRIVE,{ti},{si}")
    return JSONResponse({"ok": True})


@app.post("/api/stop")
async def api_stop():
    serial_reader.send_command("CMD,DRIVE,0,0")
    return JSONResponse({"ok": True})


@app.post("/api/emotion/{name}")
async def api_emotion(name: str):
    import tools.emotion as em
    code = em.EMOTIONS.get(name)
    if code is None:
        return JSONResponse({"error": f"unknown emotion '{name}'"}, status_code=400)
    serial_reader.send_command(f"CMD,EMOTION,{code}")
    state_bus.set_current_emotion(name)
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
