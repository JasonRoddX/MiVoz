from pathlib import Path
import argparse
import io
import sys
import time
import types

import numpy as np
import soundfile as sf
import torch

BASE = Path(__file__).resolve().parent
REPO = BASE / "Vanta-main"
sys.path.insert(0, str(REPO))

parser = argparse.ArgumentParser()
parser.add_argument("mezcla", type=Path)
parser.add_argument("--referencia", type=Path, default=BASE / "mi_voz.wav")
parser.add_argument("--sin-ganancia", action="store_true")
args = parser.parse_args()

source = (REPO / "vanta/inference.py").read_text(encoding="utf-8")
source = source.replace(
    'with torch.autocast(device_type="cuda", dtype=torch.bfloat16):',
    'with torch.autocast(device_type="cuda", enabled=False):',
)

if args.sin_ganancia:
    for line in (
        "        estimate = _for_playback(estimate)",
        "        residue = _for_playback(residue)",
    ):
        if line not in source:
            raise RuntimeError("Codigo de Vanta inesperado: " + line.strip())
        source = source.replace(line, "        pass", 1)

module = types.ModuleType("vanta_local")
sys.modules[module.__name__] = module
exec(compile(source, str(REPO / "vanta/inference.py"), "exec"), module.__dict__)

voice, rate = sf.read(args.referencia, dtype="float32")
if voice.ndim > 1:
    voice = voice.mean(axis=1)
if not np.isfinite(voice).all():
    raise ValueError("La referencia contiene valores invalidos.")

window = 5 * rate
if len(voice) < window:
    raise ValueError("La referencia debe durar al menos 5 segundos.")

starts = range(0, len(voice) - window + 1, max(1, rate // 2))
start = max(starts, key=lambda i: float(np.mean(voice[i:i+window] ** 2)))
reference = io.BytesIO()
sf.write(reference, voice[start:start+window], rate,
         format="WAV", subtype="PCM_16")

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Cargando modelo en", device, flush=True)
model = module.VantaInference(
    checkpoint_path=REPO / "checkpoints/fully_ours/best.pt",
    speaker_encoder_ckpt=str(REPO / "checkpoints/spk_encoder/best.pt"),
    device=device,
)

if device == "cuda":
    torch.cuda.synchronize()
t0 = time.perf_counter()
extracted, residue, meta = model.extract(
    args.mezcla.read_bytes(), reference.getvalue()
)
if device == "cuda":
    torch.cuda.synchronize()

suffix = "_sin_ganancia" if args.sin_ganancia else "_aislada"
output = args.mezcla.with_stem(args.mezcla.stem + suffix)
removed = args.mezcla.with_stem(args.mezcla.stem + suffix + "_retirado")
output.write_bytes(extracted)
removed.write_bytes(residue)
print(f"Tiempo: {time.perf_counter() - t0:.2f} s")
print("Salida:", output)
print("Retirado:", removed)
print("Datos:", meta)
