# MiVoz

Prototipo local en Python para aislar la voz de un hablante mediante una grabación de referencia.

Objetivo: utilizar la voz procesada como micrófono en reuniones. La implementación y las pruebas actuales se realizaron en Windows; Linux sigue pendiente.

## Estado actual

- Captura de audio con sounddevice.
- Referencia de voz utilizada para identificar al hablante mediante ECAPA.
- Separación de voces con SepFormer Libri2Mix a 8 kHz.
- En la grabación de prueba, el procesamiento completo eliminó perceptiblemente la voz de fondo y conservó la voz principal clara.
- La salida de 8 kHz tiene un timbre algo apagado.
- La prueba con ventanas de 10 segundos conservó buena separación según la evaluación auditiva.
- Las ventanas de 2 y 5 segundos dejaron pasar voz de fondo.
- El umbral de similitud mejoró los silencios, pero no eliminó los residuos durante el habla.
- VB-CABLE descargado; reinicio e integración pendientes.
- Todavía no se validó el uso como micrófono en reuniones en vivo.

## Entorno principal probado

- Windows.
- NVIDIA GeForce RTX 2080 SUPER.
- Python 3.13.17.
- PyTorch 2.6.0+cu126.
- TorchAudio 2.6.0+cu126.
- SpeechBrain y sounddevice.

Las dependencias se registran en requirements-sepformer-windows.txt.
El entorno local utilizado es .venv313.

## Scripts principales

| Archivo | Uso |
| --- | --- |
| comparar_sepformer.py | Separa una grabación completa y selecciona la voz mediante ECAPA. |
| probar_bloques10s.py | Prueba ventanas superpuestas de 10 segundos con actualizaciones cada segundo. |
| procesar.py | Prototipo anterior basado en Vanta. |

Los scripts de prueba utilizan rutas locales y la grabación de ensayo indicada en su código.

## Modelos utilizados

- Separación: https://huggingface.co/speechbrain/sepformer-libri2mix
- Identificación: https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb

ECAPA compara las pistas separadas con la voz de referencia; no realiza la separación.

## Experimentos anteriores

- Vanta: conservó la voz principal, pero dejó residuos de la voz de fondo.
- SepFormer WHAMR a 16 kHz: dejó residuos en la grabación evaluada y no se seleccionó como base.
- TSE Conv-TasNet a 48 kHz: no pudo evaluarse porque la descarga del modelo devolvió HTTP 401.

Los resultados corresponden a una prueba local con una voz de WhatsApp reproducida desde un celular detrás del hablante. No garantizan el mismo comportamiento en otros ambientes.

## Próximos pasos

1. Completar la instalación de VB-CABLE y verificar los dispositivos después del reinicio.
2. Conectar captura, procesamiento y salida virtual.
3. Medir el retraso total, la continuidad y la calidad con audio en vivo.
4. Validar el funcionamiento en una reunión de prueba.

El tiempo de procesamiento de un archivo no equivale a la latencia en vivo. La configuración de 10 segundos requiere almacenamiento temporal de audio y todavía no se considera adecuada para reuniones interactivas.

## Código de origen

Vanta: https://github.com/Komalpreet2809/Vanta

Vanta-main contiene código descargado de ese proyecto. En el prototipo se desactivó autocast bfloat16 para realizar pruebas en float32.

SpeechBrain: https://github.com/speechbrain/speechbrain

## Archivos excluidos

No se versionan entornos virtuales, grabaciones, pesos de modelos, cachés ni credenciales.

Los modelos de SpeechBrain se descargan al ejecutar los scripts y se guardan en la carpeta local modelos.
