"""LipScribe Mode A web demo (W3-10): upload or record a clip, get a caption.

    python -m lipscribe.app.demo   # P and B1 from checkpoints/
    python -m lipscribe.app.demo --checkpoint <best.pt> --decode greedy --port 7861 --share

Then open http://127.0.0.1:7860 in a browser.

Inputs: a video (upload, or record with the webcam and microphone), an optional
separate audio file or microphone recording, or a GRID test clip picked from the
manifest (uses its 16 kHz audio and speech start time, exactly like evaluate.py).
Outputs: the caption beside the video, a strip of mouth crops, and details.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from lipscribe.app.inference import FriendlyError, LipScribe, mouth_strip

TITLE = "LipScribe: lip-assisted live captioning (Mode A)"


def grid_test_clips(manifest: str | Path, splits: str | Path, limit: int = 300) -> dict[str, dict]:
    """Usable test-speaker utterances from the manifest, keyed by a readable label."""
    manifest, splits = Path(manifest), Path(splits)
    if not manifest.exists() or not splits.exists():
        return {}
    from lipscribe.data.dataset import read_splits
    part = read_splits(splits)
    clips = {}
    with open(manifest, newline="") as fh:
        for r in csv.DictReader(fh):
            if r.get("usable") == "True" and part.get(r["speaker"]) == "test" \
                    and Path(r.get("video_path", "")).exists():
                clips[f"{r['utt_id']}: \"{r['transcript']}\""] = r
                if len(clips) >= limit:
                    break
    return clips


NOISE_CHOICES = {"Clean": None, "10 dB": 10.0, "5 dB": 5.0, "0 dB": 0.0, "-5 dB": -5.0}
GATE_NOTE = "1 = trusts sound, 0 = trusts lips"
B1_LABEL = "Audio only (B1)"
P_LABEL = "Audio + lips (P)"


def gate_figure(gate, message: str | None = None):
    """Per-frame audio weight g_t against time (25 fps) as a matplotlib figure."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    fig = Figure(figsize=(7, 2.4))
    ax = fig.subplots()
    if gate is None or len(gate) == 0:
        ax.text(0.5, 0.5, message or "This model has no fusion gate (audio only).",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
    else:
        t = np.arange(len(gate)) / 25.0
        ax.plot(t, np.clip(gate, 0.0, 1.0), color="tab:blue")
        ax.set_ylim(0, 1)
        ax.set_xlim(0, max(t[-1], 0.04))
        ax.set_xlabel("time (s)")
        ax.set_ylabel("audio weight g(t)")
        ax.set_title(f"Fusion gate: {GATE_NOTE}", fontsize=10)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def build_demo(engine: LipScribe, clips: dict[str, dict] | None = None,
               compare: LipScribe | None = None, noise_splits: str | Path | None = None):
    """engine is the proposed model P; compare (optional) is the audio-only B1 for side-by-side captions."""
    import gradio as gr

    clips = clips or {}
    models = {P_LABEL: engine}
    if compare is not None:
        models = {B1_LABEL: compare, P_LABEL: engine}
    both = len(models) == 2

    def run(video, audio, clip_label, noise_label, main_model):
        snr = NOISE_CHOICES.get(noise_label)
        try:
            kw = {}
            if clip_label:
                r = clips[clip_label]
                args = (r["video_path"], r["audio_path"])
                kw = dict(speech_start_s=r.get("speech_start_s"), n_frames=int(r["n_frames"]))
                shown_video = r["video_path"]
                reference = f"\n\n**Reference:** {r['transcript']}"
            else:
                args, shown_video, reference = (video, audio), video, ""
            results = {label: m.transcribe(*args, snr_db=snr, noise_splits=noise_splits, **kw)
                       for label, m in models.items()}
        except FriendlyError as err:
            return (None, f"⚠️ {err}", "", "", None, gate_figure(None, "No result."), "")
        except Exception as err:  # keep the page alive; show a short message
            return (None, f"⚠️ Something went wrong: {type(err).__name__}: {err}", "", "", None,
                    gate_figure(None, "No result."), "")

        def text(res):
            return res.caption or "(no speech recognised)"

        main = results.get(main_model) or results[P_LABEL]
        p_res = results[P_LABEL]
        gate_fig = gate_figure(p_res.gate, "This model has no fusion gate (audio only)."
                               if p_res.gate is None else None)
        face = "n/a" if main.face_fraction is None else f"{main.face_fraction:.0%}"
        lines = [f"**Noise:** {noise_label}  \n**Frames:** {main.n_frames} ({main.n_frames / 25:.2f} s)  \n"
                 f"**Face found in:** {face} of frames"]
        for label, res in results.items():
            t = res.timings
            lines.append(f"**{label}:** {models[label].name}; preprocess {t.get('preprocess_s', 0):.2f} s, "
                         f"model {t.get('model_s', 0):.2f} s, decode {t.get('decode_s', 0):.2f} s "
                         f"(total {res.seconds:.2f} s)")
        details = "  \n".join(lines) + reference
        warns = list(dict.fromkeys(w for res in results.values() for w in res.warnings))
        if warns:
            details += "\n\n" + "\n".join(f"⚠️ {w}" for w in warns)
        return (shown_video, text(main),
                text(results[B1_LABEL]) if both else "", text(p_res),
                mouth_strip(main.crops), gate_fig, details)

    with gr.Blocks(title="LipScribe") as demo:
        gr.Markdown(f"# {TITLE}\nUpload a short clip of one speaker facing the camera, or record one. "
                    "The caption appears beside the video.")
        with gr.Row():
            with gr.Column():
                video_in = gr.Video(label="Video (upload or webcam)", sources=["upload", "webcam"],
                                    include_audio=True)
                audio_in = gr.Audio(label="Audio (optional: if the video has no sound)",
                                    sources=["upload", "microphone"], type="filepath")
                clip_in = gr.Dropdown(choices=[""] + list(clips), value="",
                                      label="…or pick a GRID test clip (same input as evaluate.py)",
                                      visible=bool(clips))
                noise_in = gr.Dropdown(choices=list(NOISE_CHOICES), value="Clean",
                                       label="Background noise (MUSAN, signal-to-noise ratio)")
                model_in = gr.Radio(choices=list(models), value=P_LABEL, label="Main caption from",
                                    visible=both)
                button = gr.Button("Caption it", variant="primary")
            with gr.Column():
                video_out = gr.Video(label="Clip", interactive=False)
                caption_out = gr.Textbox(label="Caption", lines=2)
                with gr.Row(visible=both):
                    b1_out = gr.Textbox(label=B1_LABEL, lines=2)
                    p_out = gr.Textbox(label=P_LABEL, lines=2)
                strip_out = gr.Image(label="Mouth crops (12 evenly spaced frames)", interactive=False)
                gr.Markdown(f"**Fusion gate** ({GATE_NOTE})")
                gate_out = gr.Plot(label="Audio weight per frame")
                details_out = gr.Markdown()
        button.click(run, inputs=[video_in, audio_in, clip_in, noise_in, model_in],
                     outputs=[video_out, caption_out, b1_out, p_out, strip_out, gate_out, details_out])
    return demo


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, default=Path("checkpoints/p_proposed_best.pt"),
                   help="the proposed audio + lips model P (default %(default)s)")
    p.add_argument("--compare-checkpoint", type=Path, default=Path("checkpoints/b1_audio_only_best.pt"),
                   help="audio-only model B1 for side-by-side captions; optional (default %(default)s)")
    p.add_argument("--noise-splits", type=Path, default=None,
                   help="noise split CSV (default Data/Noise_splits/splits.csv)")
    p.add_argument("--decode", choices=["beam", "greedy", "grammar"], default="beam")
    p.add_argument("--beam-width", type=int, default=10)
    p.add_argument("--device", default=None)
    p.add_argument("--manifest", default="data/manifest.csv")
    p.add_argument("--splits", default="data/splits.csv")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--share", action="store_true", help="also create a temporary public link")
    args = p.parse_args(argv)

    engine = LipScribe(args.checkpoint, device=args.device, decode=args.decode,
                       beam_width=args.beam_width)
    compare = None
    if args.compare_checkpoint and Path(args.compare_checkpoint).exists():
        compare = LipScribe(args.compare_checkpoint, device=args.device, decode=args.decode,
                            beam_width=args.beam_width)
    else:
        print(f"Comparison checkpoint {args.compare_checkpoint} not found; showing one model only.")
    clips = grid_test_clips(args.manifest, args.splits)
    print(f"Loaded {engine.name} on {engine.device}"
          + (f" and {compare.name}" if compare else "") + f"; {len(clips)} GRID test clips available")
    build_demo(engine, clips, compare=compare, noise_splits=args.noise_splits).launch(server_name="0.0.0.0", server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
