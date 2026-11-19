"""Send finished caption lines into a Zoom meeting as its own closed captions (third-party CC).

The host enables *Closed captioning* and *Allow use of caption API token* in the Zoom web portal,
then copies the meeting's caption API token (CC menu -> "Copy the API token"), a URL such as
``https://wmcc.zoom.us/closedcaption?id=...&ns=...&expire=...``. Each line is sent as
``POST <url>&seq=<n>&lang=en-US`` with a text/plain body; ``seq`` must keep rising, so the first
number continues after the token's last one (``GET .../closedcaption/seq?...``).

Only https URLs on zoom.us are accepted, so the backend cannot be made to post anywhere else.
Lines are sent from a background thread in order; a failed post is retried once and reported.
"""
from __future__ import annotations

import queue
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Callable

HTTP_HELP = {
    400: "Zoom refused the line (400): copy a fresh caption API token",
    401: "Zoom refused the caption link (401): copy the API token again from the meeting's CC menu",
    403: "Zoom refused the caption link (403): it is wrong or expired, or third-party captions are off for this meeting",
    404: "Zoom does not know this caption link (404): the meeting may have ended or restarted; copy a new token",
}


class ZoomURLError(ValueError):
    pass


def check_url(url: str) -> str:
    url = (url or "").strip()
    p = urllib.parse.urlparse(url)
    host = (p.hostname or "").lower()
    if p.scheme != "https" or not (host == "zoom.us" or host.endswith(".zoom.us")):
        raise ZoomURLError("the caption link must be a Zoom caption API token (https://….zoom.us/closedcaption?…)")
    if "closedcaption" not in p.path:
        raise ZoomURLError("that is not a Zoom caption API token: it should contain /closedcaption")
    return url


def _with(url: str, **params) -> str:
    return url + ("&" if "?" in url else "?") + urllib.parse.urlencode(params)


class ZoomCaptioner(threading.Thread):
    """Queue lines with ``send``; ``report(dict)`` gets {"type": "zoom", ...} status messages."""

    def __init__(self, url: str, report: Callable[[dict], None], lang: str = "en-US", timeout: float = 10.0):
        super().__init__(daemon=True)
        self.url, self.report, self.lang, self.timeout = check_url(url), report, lang, timeout
        self.q: queue.Queue[str | None] = queue.Queue()
        self.seq = 0
        self.sent = 0

    def send(self, line: str) -> None:
        line = " ".join((line or "").split())
        if line:
            self.q.put(line)

    def close(self) -> None:
        self.q.put(None)

    def _last_seq(self) -> int:
        p = urllib.parse.urlparse(self.url)
        seq_url = _with(urllib.parse.urlunparse(p._replace(path=p.path.rstrip("/") + "/seq")), lang=self.lang)
        try:
            with urllib.request.urlopen(seq_url, timeout=self.timeout) as r:   # check_url allows zoom.us only
                return int(r.read().decode().strip() or 0)
        except (urllib.error.URLError, ValueError, OSError):
            return 0

    def _post(self, line: str) -> None:
        req = urllib.request.Request(_with(self.url, seq=self.seq, lang=self.lang), data=line.encode("utf-8"),
                                     method="POST", headers={"Content-Type": "text/plain; charset=utf-8"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:                # check_url allows zoom.us only
            r.read()

    def run(self) -> None:
        self.seq = self._last_seq()
        while True:
            line = self.q.get()
            if line is None:
                return
            self.seq += 1
            err = None
            for attempt in range(2):
                try:
                    self._post(line)
                    err = None
                    break
                except urllib.error.HTTPError as e:
                    err = HTTP_HELP.get(e.code, f"Zoom answered {e.code}")
                    if e.code in HTTP_HELP:
                        break                              # retrying will not change Zoom's answer
                except (urllib.error.URLError, OSError) as e:
                    err = f"could not reach Zoom: {e}"
                time.sleep(1.0 + attempt)
            if err:
                self.report({"type": "zoom", "ok": False, "seq": self.seq, "message": err[:200]})
            else:
                self.sent += 1
                self.report({"type": "zoom", "ok": True, "seq": self.seq, "sent": self.sent})
