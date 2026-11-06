from pathlib import Path
import time
import numpy as np
import soundfile as sf
import soxr
import torch
from speechbrain.inference.separation import SepformerSeparation
from speechbrain.inference.speaker import EncoderClassifier
from speechbrain.utils.fetching import LocalStrategy

base = Path(r"J:\MiVoz")
original = base / "prueba_mezcla_20261106_020229.wav"
device = "cuda:0"

def load_audio(path, target_rate):
    audio, rate = sf.read(path, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if not len(audio) or not np.isfinite(audio).all():
        raise ValueError(f"Audio vacio o invalido: {path}")
    if rate != target_rate:
        audio = soxr.resample(audio, rate, target_rate)
    return np.asarray(audio, dtype=np.float32)

print("Cargando modelos...", flush=True)
separator = SepformerSeparation.from_hparams(
    source="speechbrain/sepformer-libri2mix",
    savedir=str(base / "modelos/sepformer"),
    run_opts={"device": device},
    local_strategy=LocalStrategy.COPY,
)
encoder = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir=str(base / "modelos/ecapa_voxceleb"),
    run_opts={"device": device},
    local_strategy=LocalStrategy.COPY,
)

rate = int(separator.hparams.sample_rate)
mixture = load_audio(original, rate)
reference = load_audio(base / "mi_voz.wav", 16000)

# Selecciona cinco segundos con buena energia de la referencia.
window = 5 * 16000
if len(reference) < window:
    raise ValueError("La referencia debe durar al menos cinco segundos.")
starts = range(0, len(reference) - window + 1, 8000)
start = max(starts, key=lambda i: float(np.mean(reference[i:i+window] ** 2)))
reference = reference[start:start+window]

print("Separando las dos voces...", flush=True)
torch.cuda.synchronize()
t0 = time.perf_counter()

with torch.inference_mode():
    mix_tensor = torch.from_numpy(mixture).unsqueeze(0).to(device)
    separated = separator.separate_batch(mix_tensor)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    tracks = separated[0].cpu().numpy()

    ref_tensor = torch.from_numpy(reference.copy()).unsqueeze(0).to(device)
    ref_embedding = encoder.encode_batch(ref_tensor).reshape(1, -1)

    scores = []
    for index in range(tracks.shape[1]):
        track = np.asarray(tracks[:, index], dtype=np.float32).copy()
        if rate != 16000:
            track = soxr.resample(track, rate, 16000)

        # Compara en el tramo donde hablaste junto al fondo.
        segment = track[5*16000:15*16000]
        if not len(segment):
            raise ValueError("La grabacion no contiene el tramo de comparacion.")
        tensor = torch.from_numpy(segment.copy()).unsqueeze(0).to(device)
        embedding = encoder.encode_batch(tensor).reshape(1, -1)
        score = torch.nn.functional.cosine_similarity(
            ref_embedding, embedding
        ).item()
        scores.append(score)

selected = int(np.argmax(scores))
for index in range(tracks.shape[1]):
    path = original.with_stem(original.stem + f"_sepformer_pista{index+1}")
    sf.write(path, tracks[:, index], rate, subtype="FLOAT")
    print(f"Pista {index+1}: similitud {scores[index]:.3f}")

output = original.with_stem(original.stem + "_sepformer")
sf.write(output, tracks[:, selected], rate, subtype="FLOAT")
print(f"Pista seleccionada: {selected+1}")
print(f"Separacion: {elapsed:.2f} segundos")
print(f"Resultado: {output}")