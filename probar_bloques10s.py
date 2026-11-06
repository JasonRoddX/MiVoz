from pathlib import Path
import time
import numpy as np
import soundfile as sf
import soxr
import torch
from speechbrain.inference.separation import SepformerSeparation
from speechbrain.inference.speaker import EncoderClassifier
from speechbrain.utils.fetching import LocalStrategy

BASE = Path(__file__).resolve().parent
DEVICE = "cuda:0"
ORIGINAL = BASE / "prueba_mezcla_20261106_020229.wav"

def load(path, rate):
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if sr != rate:
        audio = soxr.resample(audio, sr, rate)
    if not len(audio) or not np.isfinite(audio).all():
        raise ValueError("Audio vacio o invalido")
    return np.asarray(audio, dtype=np.float32)

print("Cargando modelos...", flush=True)
separator = SepformerSeparation.from_hparams(
    source="speechbrain/sepformer-libri2mix",
    savedir=str(BASE / "modelos/sepformer"),
    run_opts={"device": DEVICE},
    local_strategy=LocalStrategy.COPY,
)
encoder = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir=str(BASE / "modelos/ecapa_voxceleb"),
    run_opts={"device": DEVICE},
    local_strategy=LocalStrategy.COPY,
)

rate = int(separator.hparams.sample_rate)
audio = load(ORIGINAL, rate)
reference = load(BASE / "mi_voz.wav", 16000)
size = 5 * 16000
if len(reference) < size:
    raise ValueError("La referencia debe durar al menos cinco segundos.")

starts = range(0, len(reference) - size + 1, 8000)
start = max(starts, key=lambda i: float(np.mean(reference[i:i+size] ** 2)))

@torch.inference_mode()
def embedding(audio16):
    tensor = torch.from_numpy(audio16.copy()).unsqueeze(0).to(DEVICE)
    return encoder.encode_batch(tensor).reshape(1, -1)

ref_embedding = embedding(reference[start:start+size])

window = 10 * rate
hop = rate
# Superposicion y fundido para suavizar las uniones.
weight = np.hanning(window).astype(np.float32)
weight = np.maximum(weight, 0.001)
result = np.zeros(len(audio) + window, dtype=np.float32)
weights = np.zeros_like(result)
times = []

print("Procesando ventanas de 10 s cada segundo...", flush=True)
for number, position in enumerate(range(0, len(audio), hop)):
    chunk = audio[position:position+window]
    valid = len(chunk)
    chunk = np.pad(chunk, (0, window-valid))

    torch.cuda.synchronize()
    tick = time.perf_counter()
    with torch.inference_mode():
        tensor = torch.from_numpy(chunk).unsqueeze(0).to(DEVICE)
        tracks = separator.separate_batch(tensor)[0].cpu().numpy()
        scores = []
        for index in range(tracks.shape[1]):
            candidate = tracks[:, index].copy()
            candidate16 = soxr.resample(candidate, rate, 16000)
            emb = embedding(candidate16)
            score = torch.nn.functional.cosine_similarity(
                ref_embedding, emb
            ).item()
            scores.append(score)
        selected = tracks[:, int(np.argmax(scores))]
    torch.cuda.synchronize()
    times.append(time.perf_counter()-tick)

    result[position:position+window] += selected * weight
    weights[position:position+window] += weight

    if number % 8 == 0:
        print(f"Avance: {min(position+valid, len(audio))/rate:.1f} s", flush=True)

result = result[:len(audio)] / np.maximum(weights[:len(audio)], 1e-8)
if not np.isfinite(result).all():
    raise RuntimeError("La salida contiene valores invalidos.")

output = ORIGINAL.with_stem(ORIGINAL.stem + "_bloques10s")
sf.write(output, result, rate, subtype="FLOAT")

print("Guardado:", output)
print(f"Tiempo medio por ventana: {np.mean(times)*1000:.1f} ms")
print(f"Tiempo p95 por ventana: {np.percentile(times,95)*1000:.1f} ms")
print(f"Tiempo maximo: {max(times)*1000:.1f} ms")
print("Presupuesto por actualizacion: 1000 ms")
print("Esta prueba no captura el microfono ni crea un dispositivo virtual.")
