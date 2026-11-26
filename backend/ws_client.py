"""Drive the demo backend (backend/server.py) the way the page does, for tests and latency checks.

  python backend/ws_client.py "ws://127.0.0.1:8800/ws/live?code=CODE" clip grid clip.webm
  python backend/ws_client.py "ws://127.0.0.1:8800/ws/live?code=CODE" clip any clip.webm --snr -5
  python backend/ws_client.py "ws://127.0.0.1:8800/ws/live?code=CODE" live any talk.webm --realtime
  python backend/ws_client.py "ws://127.0.0.1:8800/ws/live?code=CODE" upload any film.mp4

live/clip send the file in 1 s pieces like MediaRecorder (--realtime: one piece a second), then
"stop"; upload answers the backend's "need" messages. Prints every message with its arrival time.
Needs `pip install websockets`.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

import websockets


async def run(url: str, run_: str, mode: str, path: str, snr: float | None, realtime: bool, pieces: int) -> None:
    data = Path(path).read_bytes()
    t0 = time.time()
    async with websockets.connect(url, max_size=None) as ws:
        start = {"type": "start", "run": run_, "mode": mode, "chunk_ms": 1000, "fps": 25, "snr": snr,
                 "mime": "video/mp4" if path.endswith((".mp4", ".mov")) else "video/webm",
                 "clip_seconds": 3 if run_ == "clip" else None}
        if run_ == "upload":
            start.update(size=len(data), name=path.rsplit("/", 1)[-1])
        await ws.send(json.dumps(start))

        async def feed() -> None:
            for i in range(pieces):
                await ws.send(data[i * len(data) // pieces:(i + 1) * len(data) // pieces])
                await asyncio.sleep(1.0 if realtime else 0.05)
            await ws.send(json.dumps({"type": "stop"}))

        feeder = asyncio.create_task(feed()) if run_ != "upload" else None
        n_cues = 0
        async for raw in ws:
            m = json.loads(raw)
            if m["type"] == "need":
                off = m["offset"]
                await ws.send(off.to_bytes(8, "big") + data[off:off + m["length"]])
                continue
            if m["type"] == "cues":
                n_cues += len(m["cues"])
                c = m["cues"][-1]
                print(f"[{time.time() - t0:6.1f}s] +{len(m['cues'])} cues, captioned to {m['captioned_s']}/{m['duration_s']} s;"
                      f" last {c['start']}-{c['end']} lips_used={c['lips_used']}: {c['lips']}")
                continue
            brief = {k: (f"<{len(v)}>" if k in ("gate", "crops") else v) for k, v in m.items()}
            print(f"[{time.time() - t0:6.1f}s]", json.dumps(brief))
            if m["type"] in ("final", "error"):
                break
        if feeder:
            feeder.cancel()
        if run_ == "upload":
            print("cues:", n_cues)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("url")
    ap.add_argument("run", choices=["live", "clip", "upload"])
    ap.add_argument("mode", choices=["grid", "any"])
    ap.add_argument("file")
    ap.add_argument("--snr", type=float, default=None)
    ap.add_argument("--realtime", action="store_true", help="live/clip: send one piece per second")
    ap.add_argument("--pieces", type=int, default=8)
    a = ap.parse_args()
    asyncio.run(run(a.url, a.run, a.mode, a.file, a.snr, a.realtime, a.pieces))


if __name__ == "__main__":
    main()
