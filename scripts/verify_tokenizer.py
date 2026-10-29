#!/usr/bin/env python3
"""Verify tokenizer + transcript builder against real GRID .align files."""
import sys
from pathlib import Path
from collections import Counter

sys.path.insert(0, ".")
from lipscribe.data.transcripts import parse_align
from lipscribe.data.tokenizer import encode, decode, BLANK_ID

<<<<<<< Updated upstream
ALIGN_DIR = Path("data/raw/grid")
=======
ALIGN_DIR = Path("/Users/apple/Desktop/cv_project_LipScribe/aligns")
>>>>>>> Stashed changes
EXPECTED_SLOTS = {"command": 4, "colour": 4, "preposition": 2,
                  "letter": 19, "digit": 10, "adverb": 3}  # GRID grammar
DIGITS = {"zero","one","two","three","four","five","six","seven","eight","nine"}

files = sorted(ALIGN_DIR.rglob("*.align"))
print(f"Found {len(files)} .align files\n")

bad, words_seen, longest, max_repeats = [], Counter(), (0, ""), (0, "")
for p in files:
    try:
        t = parse_align(p)
    except Exception as e:
        bad.append((p, f"parse error: {e}")); continue

    # check 1: exactly six words
    ws = t.split()
    if len(ws) != 6:
        bad.append((p, f"{len(ws)} words: {t!r}")); continue

    # check 2: round-trip through your tokenizer
    ids = encode(t)
    if decode(ids) != t:
        bad.append((p, "round-trip failed")); continue
    if BLANK_ID in ids:
        bad.append((p, "blank leaked into encode")); continue

    # check 3: grammar slots — letter single char, digit spelled out
    if len(ws[3]) != 1 or not ws[3].isalpha():
        bad.append((p, f"letter slot odd: {ws[3]!r}"))
    if ws[4] not in DIGITS:
        bad.append((p, f"digit slot odd: {ws[4]!r}"))

    words_seen.update(ws)
    if len(ids) > longest[0]:
        longest = (len(ids), t)
    r = sum(1 for a, b in zip(ids, ids[1:]) if a == b)  # adjacent repeats for CTC check
    if r > max_repeats[0]:
        max_repeats = (r, t)

print(f"Pass/fail: {len(files)-len(bad)} ok, {len(bad)} bad")
for p, why in bad[:10]:
    print("  BAD:", p.name, why)
print(f"\nUnique words observed: {len(words_seen)}")
print(sorted(words_seen))
print(f"\nLongest transcript: {longest[0]} symbols  ({longest[1]!r})")
print(f"Max adjacent repeats: {max_repeats[0]}  ({max_repeats[1]!r})")
print(f"CTC check worst case: T >= L + R = {longest[0]} + {max_repeats[0]} = {longest[0]+max_repeats[0]} (T=75 per 3s clip)")