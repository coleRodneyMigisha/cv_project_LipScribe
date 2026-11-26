import importlib
import sys

LIBS = [
    "torch", "torchvision", "torchaudio", "cv2", "mediapipe", "numpy",
    "pandas", "soundfile", "librosa", "sounddevice", "sklearn", "jiwer",
    "matplotlib", "tensorboard", "yaml", "pytest", "gradio",
]

print(f"Python {sys.version.split()[0]} on {sys.platform}")
failed = []
for name in LIBS:
    try:
        mod = importlib.import_module(name)
        print(f"OK    {name:12s} {getattr(mod, '__version__', 'n/a')}")
    except Exception as exc:
        print(f"FAIL  {name:12s} {exc}")
        failed.append(name)

import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

if failed:
    print(f"\nFAILED: {', '.join(failed)}")
    sys.exit(1)
print("\nAll libraries imported.")