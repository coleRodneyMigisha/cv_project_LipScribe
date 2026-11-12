import csv
import numpy as np
from PIL import Image, ImageDraw, ImageFont

rows = [r for r in csv.DictReader(open("flix/manifest_local.csv"))
        if r["partition"] == "train" and 3 <= len(r["transcript"].split()) <= 6]
seen, pick = set(), []
for r in rows:
    if r["speaker"] in seen:
        continue
    z = np.load(f"flix/clips/{r['utt_id']}.npz")
    c, v = z["crops"], z["valid"]
    n = int(v.sum())
    if n < 40 or c[:n].std() < 20:
        continue
    seen.add(r["speaker"]); pick.append((r, c[:n]))
    if len(pick) == 5:
        break

NAMES = {"it_derry": "IT: Welcome to Derry", "malcolm_marie": "Malcolm & Marie", "quantum": "Quantum of Solace",
         "a_haunting_in_venice": "A Haunting in Venice", "the_chosen": "The Chosen", "i_robot": "I, Robot",
         "casino_royale": "Casino Royale", "his_hers": "His & Hers"}
F, S, pad, capw = 7, 96, 12, 400
W = capw + F * (S + 4) + pad
H = len(pick) * (S + pad) + pad
img = Image.new("RGB", (W, H), (252, 252, 251)); d = ImageDraw.Draw(img)
try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 17)
    small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
except OSError:
    font = small = ImageFont.load_default()
for i, (r, c) in enumerate(pick):
    y = pad + i * (S + pad)
    d.text((pad, y + 22), "“" + r["transcript"] + "”", fill=(31, 41, 55), font=font)
    d.text((pad, y + 52), NAMES.get(r["speaker"], r["speaker"].replace("_", " ").title()), fill=(91, 100, 114), font=small)
    for k, t in enumerate(np.linspace(0, len(c) - 1, F).astype(int)):
        img.paste(Image.fromarray(c[t]).convert("RGB"), (capw + k * (S + 4), y))
img.save("reports/flix/film_mouths.png", dpi=(200, 200))
print(img.size, [p[0]["speaker"] for p in pick], [p[0]["transcript"] for p in pick])
