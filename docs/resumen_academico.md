# Reconocimiento de objetos mediante color en COIL-100

## Objetivo

Comparar histogramas de color RGB, HSV y CIELAB para identificar vistas reservadas de 20 objetos físicos de COIL-100 mediante vecino más cercano. El experimento estudia identificación dentro de un conjunto conocido de objetos, bajo las condiciones de captura de esta base de datos.

## Dataset y metodología

Se seleccionaron obj1 a obj20, con 72 vistas por objeto separadas por 5°. De las 1440 imágenes, 320 se asignaron a entrenamiento y 1120 a prueba: 16 y 56 por clase, respectivamente. Las vistas de entrenamiento se distribuyeron determinísticamente alrededor de los 360° con índices `floor(i × 72 / 16)`, para `i=0..15`. El mismo inventario se utilizó en los tres espacios.

Las imágenes se leyeron con OpenCV en BGR uint8. La máscara compartida se calculó aplicando `max(B,G,R) > 40`, seleccionando el mayor componente conexo y rellenando sus huecos internos. El respaldo ya establecido reduce el umbral a la mitad del máximo observado únicamente si no queda ningún píxel. El umbral se fijó al revisar imágenes train durante la preparación y se mantuvo en la evaluación.

Se convirtió la imagen original a RGB, HSV o CIELAB y se extrajo un histograma conjunto de 8 intervalos por canal, con 512 componentes y normalización L1. Hue se trató en el rango `[0,180)`; los otros canales uint8, incluidos los de Lab codificado por OpenCV, en `[0,256)`. Un único algoritmo 1-NN comparó cada histograma con las 320 referencias train utilizando la distancia de Bhattacharyya.

## Resultados oficiales

Se realizaron 1120 predicciones por espacio, 3360 en total. Las tres matrices de confusión contienen 56 aciertos por clase y cero errores. Para cada objeto, TP=56, FP=0, TN=1064 y FN=0; todas las clases tienen soporte 56. Los promedios macro incluyen las 20 clases y toda división entre cero se define como 0.

| Espacio | Aciertos / prueba | Accuracy | Error | Precision macro | Recall macro | F1 macro |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| RGB | 1120 / 1120 | 100 % | 0 % | 100 % | 100 % | 100 % |
| HSV | 1120 / 1120 | 100 % | 0 % | 100 % | 100 % | 100 % |
| CIELAB | 1120 / 1120 | 100 % | 0 % | 100 % | 100 % | 100 % |

No se observa una ventaja de un espacio de color sobre otro bajo este protocolo. Los CSV individuales, matrices y figuras se conservan en `artifacts/phase3b/`.

## Interpretación del resultado perfecto

El protocolo permite un desempeño alto porque las consultas pertenecen a los mismos objetos físicos que las referencias. Las 16 vistas de entrenamiento cubren el giro con separaciones de 20° o 25°, de modo que las vistas reservadas tienen referencias angularmente cercanas. El fondo oscuro y las condiciones de adquisición controladas también reducen la variabilidad de apariencia. Las distribuciones de color pueden ser suficientemente distintivas para separar este subconjunto de objetos sin utilizar información de forma.

Estas condiciones ofrecen una explicación plausible del 100 % observado; el experimento no aísla causalmente la contribución de cada factor. No se evaluaron los 100 objetos completos, nuevas instancias de las clases, fondos diferentes ni cambios de iluminación.

La verificación adicional reprodujo los 3360 vecinos oficiales calculando distancias desde los histogramas, sin pasar la etiqueta de la consulta al selector. Seis imágenes renombradas de manera engañosa conservaron sus predicciones. Al permutar las etiquetas de entrenamiento con semilla 20261017, la exactitud diagnóstica cayó a 2,68 % en RGB y 2,32 % en HSV y CIELAB. Estos controles apoyan la integridad del flujo y permanecen separados de las métricas oficiales.

## Limitaciones y conclusión

La segmentación puede perder ruedas, bordes oscuros o partes desconectadas, e incluir fondo dentro de asas. Los histogramas globales no conservan la estructura espacial y pueden confundir objetos de colores parecidos. El 1-NN siempre devuelve una clase; su distancia no expresa confianza calibrada ni permite afirmar que se detectan objetos desconocidos. La división por vistas mantiene la identidad física del objeto entre entrenamiento y prueba.

El método identificó correctamente las vistas reservadas de los 20 objetos seleccionados bajo este protocolo fijo. El resultado justifica documentar la utilidad de características simples de color en un escenario controlado, con un alcance limitado al experimento realizado. Para estudiar generalización se necesitarían evaluaciones independientes bajo otras instancias y condiciones; no se realizaron como parte de esta práctica.

Fuente del dataset: [COIL-100, Columbia University](https://www.cs.columbia.edu/CAVE/software/softlib/coil-100.php).
