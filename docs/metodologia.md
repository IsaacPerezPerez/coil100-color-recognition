# Metodología y reproducción

## Organización

Los módulos están en `src/coil100/`: `dataset.py` y `prepare.py` validan COIL-100; `inventory.py` lee la división; `preprocessing.py` obtiene la máscara; `features.py` extrae histogramas; `classifier.py` implementa el 1-NN; `recognize.py` ofrece la consola; `metrics.py`, `evaluation_plots.py` y `evaluate.py` producen la evaluación.

`audit_training.py` e `inspect_training.py` permiten revisar entrenamiento localmente. `verify_independent.py` ejecuta controles adicionales y `scripts/build_review_zip.py` conserva el procedimiento de revisión privada. Los mosaicos y ZIP generados localmente no forman parte de la publicación.

## Segmentación y características

La máscara se calcula una vez por imagen a partir de `max(B,G,R) > 40`, con conectividad 8 y selección del componente de mayor área. Se rellenan sus huecos internos. Si el umbral elimina todos los píxeles se usa la mitad del máximo observado; una imagen completamente negra produce un error. Los parámetros se fijaron observando entrenamiento y no se cambiaron tras evaluar test.

La imagen original se convierte desde BGR uint8:

| Espacio | Conversión | Rangos, con límite superior excluido |
| --- | --- | --- |
| RGB | `COLOR_BGR2RGB` | R, G, B: `[0,256)` |
| HSV | `COLOR_BGR2HSV` | H: `[0,180)`; S, V: `[0,256)` |
| CIELAB | `COLOR_BGR2LAB` | L, a, b uint8: `[0,256)` |

La codificación uint8 de Lab escala L de 0–100 a 0–255 y desplaza a/b en 128. Los tres histogramas usan la misma máscara, 8 intervalos por canal y normalización por la suma. No se utilizan rangos físicos de Lab directamente para los bins.

## Clasificación y métricas

Se usa `cv2.compareHist(..., cv2.HISTCMP_BHATTACHARYYA)`. Para histogramas normalizados equivale a `sqrt(1 − sum(sqrt(p_i*q_i)))`. La clase devuelta es la del vecino con menor distancia. Los empates se resuelven ordenando las referencias por objeto, ángulo y ruta; en las 3360 predicciones oficiales no hubo empates.

TP es la diagonal, FP la suma de columna menos TP, FN la suma de fila menos TP y TN el total menos TP, FP y FN. Precision=TP/(TP+FP), Recall=TP/(TP+FN), F1=2TP/(2TP+FP+FN). Toda división entre cero devuelve 0; los promedios macro incluyen las 20 clases. Accuracy es la diagonal total dividida entre 1120, no la media de exactitudes uno contra el resto.

## Evidencias y comandos adicionales

Se incluyen resultados numéricos de auditoría y evaluación, los cuatro gráficos estadísticos propios y metadatos de integridad. Las fotografías, capturas y mosaicos de COIL-100 se omiten al no disponer de autorización clara de redistribución. Las menciones a esos archivos en notas históricas describen la revisión local; no son archivos publicados.

Con el dataset descargado, se pueden repetir los diagnósticos en carpetas nuevas:

```powershell
.\.venv\Scripts\python.exe -m src.coil100.audit_training --output-dir artifacts/phase3a_reproduction
.\.venv\Scripts\python.exe -m src.coil100.verify_independent --output-dir artifacts/phase4_reproduction --seed 20261017
```

No se necesitan imágenes originales para ejecutar las 43 pruebas de la distribución pública. Las once pruebas de resultados guardados comparan los CSV con el inventario incluido y con los hashes de los módulos. Las restantes usan datos sintéticos temporales y controles de empaquetado. Los diagnósticos reales y las inferencias sí requieren COIL-100.

Las versiones de la evaluación están registradas en `artifacts/phase3b/evaluation_metadata.json`: Python 3.13.7, NumPy 2.5.4, OpenCV 4.14.0.94, Pillow 12.3.0 y Matplotlib 3.11.2. Las figuras regeneradas pueden cambiar visualmente con otras versiones, aunque las métricas no cambien. Las dependencias directas y sus rangos están en `requirements.txt`.

Los metadatos oficiales fueron generados antes del commit público inicial y conservan `git_revision: null`; sus hashes SHA-256 identifican el código experimental de ese momento. La publicación mantiene esos archivos sin reescribir los resultados históricos.
