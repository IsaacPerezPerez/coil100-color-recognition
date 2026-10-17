# Auditoría de entrenamiento — Fase 3A

Se conservó la implementación de segmentación, extracción y clasificación de Fase 2, con umbral 40. La auditoría solo abrió imágenes train; test se consultó como metadatos del inventario. No hubo búsqueda de hiperparámetros.

## Revisión del algoritmo

- BGR se convierte mediante `COLOR_BGR2RGB`, `COLOR_BGR2HSV` o `COLOR_BGR2LAB`.
- Hue utiliza `[0, 180)`; S/V y los canales RGB/Lab uint8 utilizan `[0, 256)`.
- README documenta L escalado a uint8 y a/b desplazados 128 en CIELAB.
- Se comprobó que los 960 histogramas de entrenamiento tienen 512 componentes finitos no negativos y suma cercana a 1.
- Una misma máscara original alimenta los tres extractores. La prueba correspondiente comprueba la identidad de la máscara y que no se modifica.
- Se utiliza `cv2.compareHist(..., HISTCMP_BHATTACHARYYA)` tanto en la predicción normal como en Leave-One-Out. La prueba de solapamiento parcial verifica una distancia conocida.
- El constructor rechaza referencias test; la carga desde inventario filtra train antes de abrir imágenes. Las pruebas comprueban esta exclusión.
- La división se verificó contra la regla original y el hash del inventario coincide con el registrado en Fase 2: `0d50e33f53571f06bb7fa5a5efde24d4e3213d7a7e2793f80185a941664e5393`.

## Revisión visual

El mosaico principal presenta 0° y 180° de cada objeto: son las posiciones 0 y 8 de sus 16 vistas train ordenadas. Las 40 máscaras muestran la región principal del objeto; ninguna está vacía. El análisis numérico cubre las 320 vistas train, no solo las mostradas.

Casos que requieren atención visual, sin disponer de máscaras manuales de referencia:

- **obj6, obj8 y obj15 (automóviles), 0°/180°:** ruedas, bajos y algunas zonas oscuras de ventanas parecen recortados parcialmente o unidos al fondo. Los porcentajes bajos también reflejan su silueta alargada; no cuantifican el error de segmentación.
- **obj19 (vehículo rojo), 0°/180°:** atención a ruedas y contorno inferior oscuro, donde se pierden fragmentos de silueta.
- **obj10, obj11 y obj16 (tazas), 0°/180°:** el relleno del contorno incorpora fondo dentro de las asas. En la taza marrón obj11 se conserva el cuerpo oscuro principal, pero el hueco del asa queda incluido en la máscara.
- **obj5 y obj16, 0°:** pequeños salientes de la máscara en bordes laterales podrían corresponder a ruido de fondo conectado al objeto.

Se revisó además `train_mask_extremes.png`: las vistas de mayor cobertura de los seis objetos con mayor variación entre sus 16 máscaras (obj6, obj15, obj8, obj1, obj3 y obj19). Los porcentajes altos corresponden visualmente a vistas en las que el objeto ocupa más área; no se observa una inclusión masiva de fondo como la que produciría un umbral demasiado bajo. Este gráfico adicional se selecciona determinísticamente por rango máximo−mínimo, sin modificar parámetros.

No hay máscaras vacías ni uso del umbral de respaldo en las 320 imágenes. Esto no demuestra que todas las siluetas sean correctas. La selección del componente mayor puede perder partes desconectadas y el relleno puede introducir fondo; se mantiene esta limitación para la evaluación posterior.

## Píxeles conservados por objeto

Porcentajes respecto del total de píxeles de cada imagen; mínimo/media/máximo consideran sus 16 vistas train.

| Objeto | Mínimo % | Media % | Máximo % | 0° % | 180° % |
| --- | ---: | ---: | ---: | ---: | ---: |
| obj1 | 28.49 | 37.43 | 69.96 | 34.12 | 36.05 |
| obj2 | 62.59 | 69.27 | 76.25 | 71.04 | 73.88 |
| obj3 | 29.41 | 45.30 | 70.29 | 29.41 | 29.41 |
| obj4 | 59.86 | 62.27 | 64.56 | 63.57 | 64.56 |
| obj5 | 43.14 | 43.68 | 44.10 | 43.59 | 43.91 |
| obj6 | 22.42 | 36.53 | 86.56 | 22.42 | 25.07 |
| obj7 | 47.51 | 48.32 | 48.90 | 48.51 | 48.02 |
| obj8 | 23.56 | 36.51 | 80.36 | 24.96 | 25.29 |
| obj9 | 51.84 | 57.96 | 65.97 | 57.69 | 63.16 |
| obj10 | 54.44 | 68.10 | 80.41 | 54.44 | 55.46 |
| obj11 | 43.13 | 51.97 | 59.51 | 43.69 | 43.13 |
| obj12 | 42.07 | 54.69 | 59.85 | 58.29 | 58.42 |
| obj13 | 25.35 | 41.06 | 50.75 | 50.32 | 50.75 |
| obj14 | 42.24 | 47.29 | 54.64 | 54.64 | 49.56 |
| obj15 | 23.07 | 35.99 | 86.51 | 25.45 | 26.42 |
| obj16 | 66.39 | 70.52 | 75.78 | 70.52 | 70.99 |
| obj17 | 32.16 | 36.70 | 42.92 | 41.02 | 42.92 |
| obj18 | 66.34 | 67.24 | 68.47 | 66.80 | 66.39 |
| obj19 | 30.83 | 41.99 | 71.11 | 32.03 | 31.92 |
| obj20 | 49.16 | 51.70 | 55.28 | 50.08 | 52.09 |

## Leave-One-Out: diagnóstico sobre entrenamiento

| Espacio | Aciertos | Consultas | Exactitud |
| --- | ---: | ---: | ---: |
| RGB | 320 | 320 | 100,00 % |
| HSV | 320 | 320 | 100,00 % |
| CIELAB | 320 | 320 | 100,00 % |

Cada consulta se predijo con otras 319 imágenes train. La fila consultada se elimina antes de construir el clasificador. En las 960 predicciones guardadas no hay vecinos con la misma ruta que su consulta ni distancias cero. El resultado no equivale a exactitud final sobre imágenes no utilizadas durante el desarrollo: las referencias incluyen otras vistas del mismo objeto y los parámetros de segmentación se fijaron previamente observando entrenamiento.

No se detectaron fallos funcionales en las conversiones, los histogramas ni el cálculo del vecino. Se mantienen limitaciones del método: el histograma pierde estructura espacial; el 1-NN siempre asigna una clase y su distancia no es confianza; los empates favorecen el primer objeto según el orden determinista. No se cambió el clasificador para mejorar este diagnóstico.

## Pruebas, archivos y publicación

Las 22 pruebas automatizadas pasaron (19 existentes y 3 nuevas), incluida una comprobación sintética que detectaría la inclusión de la propia consulta en Leave-One-Out y otra que verifica la exportación JSON y la lectura exclusiva de train. El primer intento de exportación reveló un escalar NumPy no serializable: se corrigió la conversión a tipos Python en el nuevo módulo de auditoría.

Archivos añadidos: `src/coil100/audit_training.py`, `tests/test_audit_training.py` y los artefactos de `artifacts/phase3a/`. Archivos modificados: `README.md` y `.gitignore`. Se conservaron los módulos funcionales de Fases 1 y 2 y las dependencias existentes.

`.gitignore` excluye entornos, datos originales, cachés, temporales, archivos .env y configuración de editores. Los resultados y gráficos de artifacts permanecen disponibles para documentación. No se publicó el proyecto ni se ejecutó la evaluación final de las 1120 imágenes test.
