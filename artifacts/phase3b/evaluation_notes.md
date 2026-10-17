# Evaluación final — Fase 3B

Los resultados corresponden a 1120 vistas no utilizadas como referencia de los mismos 20 objetos físicos de COIL-100. No evalúan reconocimiento de objetos arbitrarios ni generalización a otras condiciones de captura.

## Metodología congelada

Se mantuvieron la división original, el umbral 40 y su respaldo ya definido, el componente mayor con relleno de huecos, una máscara compartida por imagen, los histogramas 8×8×8 normalizados L1 y el 1-NN con distancia de Bhattacharyya de OpenCV. Las únicas referencias fueron las 320 filas train. No se ajustó el método usando test.

Se ejecutaron exactamente 1120 predicciones por espacio: 3360 en total. Las tres evaluaciones compartieron imágenes, etiquetas y segmentación. Los archivos individuales conservan todos los aciertos y errores.

## Resultados

| Espacio | Aciertos / 1120 | Accuracy | Error | Precision macro | Recall macro | F1 macro |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| RGB | 1120 / 1120 | 1.000000 | 0.000000 | 1.000000 | 1.000000 | 1.000000 |
| HSV | 1120 / 1120 | 1.000000 | 0.000000 | 1.000000 | 1.000000 | 1.000000 |
| CIELAB | 1120 / 1120 | 1.000000 | 0.000000 | 1.000000 | 1.000000 | 1.000000 |

## Objetos con más errores

- RGB: ningún objeto presentó errores.
- HSV: ningún objeto presentó errores.
- CIELAB: ningún objeto presentó errores.

## Definición y verificación de métricas

Las filas de las matrices son clases reales y las columnas predichas; el orden fijo es obj1 a obj20. TP es la diagonal, FP la suma de columna menos TP, FN la suma de fila menos TP y TN el total menos TP, FP y FN. Precision=TP/(TP+FP), Recall=TP/(TP+FN), F1=2TP/(2TP+FP+FN). Toda división entre cero devuelve 0. Los promedios macro incluyen las 20 clases.

Accuracy global es la suma de la diagonal dividida entre 1120; Error=1−Accuracy. No se utiliza la media de exactitudes uno contra el resto. Las pruebas verifican conteos, identidades matemáticas y ausencia de NaN/infinito, tanto con matrices conocidas como con los archivos reales guardados.

## Integridad y reproducción

SHA-256 del inventario antes y después: `0d50e33f53571f06bb7fa5a5efde24d4e3213d7a7e2793f80185a941664e5393`.

Revisión Git: sin commit válido; se registran hashes de archivos en evaluation_metadata.json. Los módulos del método conservaron sus hashes durante toda la ejecución.

```powershell
.\.venv\Scripts\python.exe -m src.coil100.evaluate
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

El comando rechaza sobrescribir un directorio que ya contiene resultados de evaluación. Para otra reproducción usa `--output-dir artifacts/phase3b_reproduction`. Las pruebas de resultados reales leen los CSV guardados y no ejecutan predicciones adicionales.

Versiones: python 3.13.7, numpy 2.5.4, opencv-python-headless 4.14.0.94, Pillow 12.3.0, matplotlib 3.11.2.

## Limitaciones

Las máscaras pueden perder ruedas y bordes oscuros o incluir fondo dentro de asas. Los histogramas no conservan estructura espacial, el clasificador asigna siempre una clase y la distancia no es una confianza calibrada. La auditoría Leave-One-Out previa es un diagnóstico de entrenamiento independiente de esta evaluación. No se buscaron hiperparámetros ni se filtraron resultados desfavorables.

## Comprobaciones posteriores a la ejecución

Las 38 pruebas automatizadas pasaron: 22 previas, 5 de fórmulas con matrices conocidas y 11 sobre los artefactos reales. El registro completo está en `unit_tests.txt`.

Se calculó SHA-256 de los 1440 PNG seleccionados, guardado en `image_sha256.csv`: 320 hashes distintos en train, 1120 en test y cero compartidos entre conjuntos. Esta verificación adicional no realizó predicciones ni modificó el método.

Se inspeccionaron las cuatro figuras PNG: etiquetas, orientación de matrices y escala de comparación legibles. Cada clase tiene TP=56, FP=0, TN=1064 y FN=0. El resultado perfecto no elimina las limitaciones de segmentación y alcance descritas arriba.
