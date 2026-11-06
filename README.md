# MiVoz

Prototipo local para aislar la voz de un hablante usando una grabación de referencia.

## Estado actual

- Captura de audio con sounddevice.
- Referencia limpia de voz grabada.
- Inferencia de Vanta comprobada con NVIDIA RTX 2080 SUPER.
- La voz principal queda clara; persisten residuos bajos de otras voces.
- Pendiente: comparar salida sin normalización, mejorar residuos y evaluar streaming.
- Todavía no funciona como micrófono virtual para llamadas.

## Entorno probado

Windows, Python 3.15.2, torch 2.14.1+cu126 y torchaudio 2.11.1+cpu.
Se comprobó MelSpectrogram en CUDA; esto no garantiza compatibilidad de todas las funciones.

## Código de origen

Vanta: https://github.com/Komalpreet2809/Vanta

Vanta-main contiene el código descargado de ese proyecto.
El ajuste local desactiva autocast bfloat16 para probar inferencia en float32.

## Archivos excluidos

No se versionan entornos, grabaciones, modelos descargados ni cachés.
Los pesos se descargan con scripts/download_weights.py desde Vanta-main.
