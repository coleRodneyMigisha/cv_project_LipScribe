"""lipscribe.app.zoom_captions: lines go to Zoom's caption API in order (network replaced by a fake)."""
from __future__ import annotations

import io
import urllib.error
import urllib.parse

import pytest

from lipscribe.app import zoom_captions as zc

TOKEN = "https://wmcc.zoom.us/closedcaption?id=123&ns=abc&expire=86400&sparams=id%2Cns%2Cexpire&signature=x"


@pytest.mark.parametrize("bad", ["http://wmcc.zoom.us/closedcaption?id=1", "https://evil.example/closedcaption?id=1",
                                 "https://zoom.us.evil.example/closedcaption", "https://wmcc.zoom.us/other?id=1", ""])
def test_only_zoom_caption_links_are_accepted(bad):
    with pytest.raises(zc.ZoomURLError):
        zc.check_url(bad)


def test_check_url_accepts_a_token():
    assert zc.check_url(" " + TOKEN + " ") == TOKEN


class FakeZoom:
    def __init__(self, last_seq=41, fail_first=0):
        self.last_seq, self.fail_first, self.posts = last_seq, fail_first, []

    def __call__(self, req, timeout=None):
        url = req if isinstance(req, str) else req.full_url
        q = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
        if isinstance(req, str):                          # GET .../closedcaption/seq
            assert urllib.parse.urlparse(url).path == "/closedcaption/seq" and q["lang"] == ["en-US"]
            return io.BytesIO(str(self.last_seq).encode())
        if self.fail_first:
            self.fail_first -= 1
            raise urllib.error.URLError("down")
        assert req.get_method() == "POST" and req.headers["Content-type"].startswith("text/plain")
        self.posts.append((int(q["seq"][0]), q["lang"][0], req.data.decode()))
        return io.BytesIO(b"")


def run_lines(monkeypatch, fake, lines):
    monkeypatch.setattr(zc.urllib.request, "urlopen", fake)
    monkeypatch.setattr(zc.time, "sleep", lambda s: None)
    reports = []
    z = zc.ZoomCaptioner(TOKEN, reports.append)
    z.start()
    for x in lines:
        z.send(x)
    z.close()
    z.join(5)
    return reports


def test_lines_are_posted_in_order_after_the_tokens_last_seq(monkeypatch):
    fake = FakeZoom(last_seq=41)
    reports = run_lines(monkeypatch, fake, ["hello there", "  ", "how  are you"])
    assert fake.posts == [(42, "en-US", "hello there"), (43, "en-US", "how are you")]   # blank skipped
    assert [r["ok"] for r in reports] == [True, True] and reports[-1]["sent"] == 2


def test_a_failed_post_is_retried_once_then_reported(monkeypatch):
    fake = FakeZoom(last_seq=0, fail_first=3)        # line 1 fails twice -> reported; line 2: 1 fail + success
    reports = run_lines(monkeypatch, fake, ["one", "two"])
    assert reports[0]["ok"] is False and "reach Zoom" in reports[0]["message"]
    assert reports[1]["ok"] is True and fake.posts == [(2, "en-US", "two")]


def test_a_refused_token_is_explained_without_retrying(monkeypatch):
    calls = []

    def fake(req, timeout=None):
        if isinstance(req, str):
            return io.BytesIO(b"0")
        calls.append(req.full_url)
        raise urllib.error.HTTPError(req.full_url, 403, "Forbidden", {}, io.BytesIO(b"{}"))

    reports = run_lines(monkeypatch, fake, ["hello"])
    assert len(calls) == 1 and reports[0]["ok"] is False and "403" in reports[0]["message"]
