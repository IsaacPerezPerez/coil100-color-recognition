# Verificación independiente y evidencias — Fase 4

## Alcance y flujo revisado

Se inspeccionaron `recognize.py`, `classifier.py`, `features.py`, `preprocessing.py`, `inventory.py` y `evaluate.py`. La consola decodifica la ruta de consulta mediante OpenCV y entrega únicamente el arreglo BGR a `recognize`. La máscara y las conversiones dependen de los píxeles. `predict` recibe un histograma y el espacio de color, calcula 320 distancias y elige la menor. La etiqueta real de test se incorpora al registro después de predecir y se utiliza para calcular métricas.

Las etiquetas de las referencias son necesarias para asignar la clase del vecino. Además, el constructor ordena referencias por clase, ángulo y ruta para resolver empates exactos. Por tanto, no sería correcto afirmar que los metadatos nunca pueden intervenir en un desempate. La etiqueta real y el nombre de la consulta no se entregan al selector de vecino. En las 3360 comparaciones oficiales no se observaron empates ni diferencias entre las dos menores distancias inferiores a 1e-12, por lo que el desempate no afectó estos resultados.

Los parámetros se fijaron en las fases previas usando entrenamiento. La Fase 3A no abrió imágenes test y la Fase 3B conservó los hashes de los módulos. La revisión actual no encontró rutas de ajuste basadas en test. Esto describe el código y los antecedentes disponibles; no sustituye una auditoría externa exhaustiva del historial, que todavía no cuenta con commits identificables.

## Controles ejecutados

### Nombres engañosos

Se eligió la primera imagen test por ángulo para obj1, obj4, obj6, obj11, obj15 y obj19. Cada una se copió, sin cambiar sus bytes, a un directorio temporal con el nombre `obj(21−clase)__355.png`. Se ejecutó la consola real sobre el original y sobre la copia, en los tres espacios. En las 18 comparaciones coincidieron clase predicha, distancia, vecino y espacio. Las copias temporales se eliminaron al terminar. Los pares se registran en `artifacts/phase4/renamed_queries.csv` sin rutas personales.

### Distancias desde los píxeles

Se extrajeron nuevamente los histogramas con el método fijo y se calcularon directamente las distancias a las 320 referencias train. Los 3360 mínimos coincidieron con el vecino, la clase y la distancia de los CSV oficiales. Este cálculo de control no recibe la etiqueta real ni analiza el nombre de la consulta. Las menores separaciones entre primer y segundo vecino fueron aproximadamente 4,39e-5 en RGB, 5,80e-5 en HSV y 6,16e-5 en CIELAB.

### Etiquetas de entrenamiento permutadas

Se usó `numpy.random.default_rng(20261017)` para permutar las 320 etiquetas de las referencias, conservando 16 etiquetas de cada clase. Los histogramas, imágenes, parámetros y código del clasificador no se cambiaron. Las nuevas asociaciones existieron solo en memoria y se guardaron como evidencia separada. La única finalidad fue comprobar que romper la relación entre apariencia y etiqueta deteriora el desempeño.

| Espacio | Aciertos del control / 1120 | Exactitud diagnóstica |
| --- | ---: | ---: |
| RGB | 30 / 1120 | 2,6786 % |
| HSV | 26 / 1120 | 2,3214 % |
| CIELAB | 26 / 1120 | 2,3214 % |

Es una sola permutación con semilla fijada, no una estimación estadística basada en muchas repeticiones. No se eligió la semilla buscando un resultado particular. Estos valores no reemplazan los resultados oficiales de Fase 3B.

## Integridad de resultados

Los tres CSV oficiales contienen exactamente 1120 registros y todas las referencias seleccionadas proceden de train. Las matrices reconstruidas desde los CSV coinciden con las guardadas: 56 en cada celda diagonal y cero fuera de ella. Las métricas por objeto y macro se verificaron mediante fórmulas directas. La separación por rutas y los hashes de imágenes de Fase 3B no muestran archivos originales idénticos compartidos entre train y test.

Las tres matrices PNG y la comparación se regeneraron en un directorio temporal desde sus CSV; sus píxeles resultaron idénticos a los de las figuras oficiales. La regeneración no sobrescribió archivos de Fase 3B. Se verificaron los hashes de todos sus archivos y de los módulos originales antes y después del diagnóstico, además del inventario SHA-256 `0d50e33f53571f06bb7fa5a5efde24d4e3213d7a7e2793f80185a941664e5393`.

No se identificó una fuga de información que explique el resultado perfecto. Esta conclusión se limita a las rutas de inferencia, los controles y los artefactos examinados. Se conserva la limitación de generalización a los mismos objetos físicos y condiciones de captura.

## Reproducción y paquete de revisión

La suite completa aprobó **43 pruebas**: las 38 existentes, tres controles nuevos de independencia y dos de empaquetado. El registro actual está en `artifacts/phase4/unit_tests.txt`. Los resultados oficiales y su registro previo de 38 pruebas se conservaron sin cambios.

```powershell
.\.venv\Scripts\python.exe -m src.coil100.verify_independent --output-dir artifacts/phase4_reproduction --seed 20261017
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe scripts/build_review_zip.py
```

El diagnóstico requiere el dataset real original y conserva los resultados oficiales. Para los controles sobre figuras se recomienda el entorno registrado en `evaluation_metadata.json`, ya que distintas versiones pueden cambiar el renderizado sin cambiar los datos numéricos.

El ZIP incluye código, pruebas, documentación, inventario numérico, todos los CSV oficiales de predicciones y métricas, matrices, figuras, mosaico y evidencias de Fases 3A y 4. Incluye el inventario aunque el dataset original se excluye, para poder revisar los artefactos guardados sin descargar imágenes. Las pruebas sintéticas y de CSV guardados pueden ejecutarse tras instalar las dependencias; los diagnósticos reales requieren obtener COIL-100 desde Columbia University.

La selección explícita excluye datos raw, entornos, Git, cachés, temporales y artefactos históricos de Fase 2 que contienen rutas locales. Los resultados originales de Fase 2 permanecen en el workspace. El archivo `MANIFEST.json` del ZIP registra tamaño y SHA-256 de cada archivo incluido. La escritura se realiza a `.zip.partial`, se verifican CRC y contenido descomprimido y después se renombra. El informe final de integridad del ZIP se guarda aparte para evitar una referencia circular a su propio hash.

El nombre previsto del repositorio es `coil100-color-recognition`. No se publicaron archivos, no se cambió visibilidad y no se generó un PDF. Antes de la publicación quedan pendientes la revisión externa del paquete y la decisión sobre autoría, licencia y contenido definitivo.
