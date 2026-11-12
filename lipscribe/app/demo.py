"""LipScribe Mode A web demo (W3-10): upload or record a clip, get a caption.

    python -m lipscribe.app.demo --checkpoint runs/b1_audio_only_20261103-193743/best.pt
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


def build_demo(engine: LipScribe, clips: dict[str, dict] | None = None):
    import gradio as gr

    clips = clips or {}

    def run(video, audio, clip_label):
        try:
            if clip_label:
                r = clips[clip_label]
                result = engine.transcribe(r["video_path"], r["audio_path"],
                                           speech_start_s=r.get("speech_start_s"),
                                           n_frames=int(r["n_frames"]))
                shown_video = r["video_path"]
                reference = f"\n\n**Reference:** {r['transcript']}"
            else:
                result = engine.transcribe(video, audio)
                shown_video, reference = video, ""
        except FriendlyError as err:
            return None, f"⚠️ {err}", None, ""
        except Exception as err:  # keep the page alive; show a short message
            return None, f"⚠️ Something went wrong: {type(err).__name__}: {err}", None, ""

        face = "n/a" if result.face_fraction is None else f"{result.face_fraction:.0%}"
        details = (f"**Model:** {engine.name}  \n**Frames:** {result.n_frames} "
                   f"({result.n_frames / 25:.2f} s)  \n**Face found in:** {face} of frames  \n"
                   f"**Time:** {result.seconds:.1f} s{reference}")
        if result.warnings:
            details += "\n\n" + "\n".join(f"⚠️ {w}" for w in result.warnings)
        return shown_video, result.caption or "(no speech recognised)", mouth_strip(result.crops), details

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
                button = gr.Button("Caption it", variant="primary")
            with gr.Column():
                video_out = gr.Video(label="Clip", interactive=False)
                caption_out = gr.Textbox(label="Caption", lines=2)
                strip_out = gr.Image(label="Mouth crops (12 evenly spaced frames)", interactive=False)
                details_out = gr.Markdown()
        button.click(run, inputs=[video_in, audio_in, clip_in],
                     outputs=[video_out, caption_out, strip_out, details_out])
    return demo


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True, type=Path, help="e.g. runs/<b1 run>/best.pt")
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
    clips = grid_test_clips(args.manifest, args.splits)
    print(f"Loaded {engine.name} on {engine.device}; {len(clips)} GRID test clips available")
    build_demo(engine, clips).launch(server_name="0.0.0.0", server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
