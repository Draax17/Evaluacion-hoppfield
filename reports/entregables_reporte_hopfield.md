# Entregables para el reporte

## Proyecto: Evaluación experimental de Redes de Hopfield como Memoria Asociativa

Este documento contiene texto base para integrar en el reporte final de la asignación. Está organizado de acuerdo con los entregables solicitados: descripción detallada del procedimiento, gráficas de resultados para ambas redes, análisis comparativo y conclusiones sobre la capacidad de almacenamiento.

---

## 1. Descripción detallada del procedimiento seguido

El objetivo de la práctica fue analizar de forma experimental la capacidad de almacenamiento y recuperación de patrones de una Red de Hopfield, utilizando letras del alfabeto codificadas como imágenes binarias. Además, el desempeño de esta red se compara con una Memoria Hebbiana autoasociativa clásica sin dinámica de actualización.

Para ello, se utilizó un conjunto de datos compuesto por las letras de la A a la Z en tres tamaños diferentes de imagen: 7x7, 10x10 y 15x15 pixeles. Cada imagen fue transformada a un patrón bipolar, donde los pixeles pertenecientes a la letra se representan con el valor +1 y los pixeles de fondo se representan con el valor -1. Esta representación permite que los patrones sean usados directamente por modelos de memoria asociativa.

El procedimiento general se dividió en las siguientes etapas:

1. **Preparación del conjunto de datos.**  
   Se generaron o cargaron los archivos `.npz` correspondientes a cada tamaño de imagen. Cada archivo contiene las matrices de las letras, sus vectores aplanados y sus etiquetas. Para cada tamaño se trabajó con 26 patrones, uno por cada letra del alfabeto.

2. **Entrenamiento de la Red de Hopfield.**  
   La Red de Hopfield se entrenó utilizando la regla de aprendizaje Hebbiana simétrica. A partir de un conjunto de patrones almacenados, se construyó una matriz de pesos cuadrada. La diagonal principal de esta matriz se estableció en cero para eliminar autoconexiones, ya que una neurona no debe reforzarse a sí misma.

3. **Recuperación de patrones con Hopfield.**  
   Una vez entrenada la red, se le presentó un patrón inicial, que puede ser limpio o tener ruido. La red actualizó iterativamente su estado hasta alcanzar convergencia o hasta llegar a un número máximo de iteraciones. El patrón recuperado se comparó contra los patrones originales para determinar si la recuperación fue correcta.

4. **Evaluación de capacidad de almacenamiento.**  
   Para cada tamaño de imagen, se entrenó la red almacenando progresivamente de 1 a 26 letras. Después de cada entrenamiento, se evaluó cuántas letras eran recuperadas correctamente. Esto permitió observar en qué punto la red comenzaba a fallar al aumentar el número de patrones almacenados.

5. **Evaluación de tolerancia al ruido.**  
   Para medir la robustez de la red, se alteró aleatoriamente un porcentaje de bits en los patrones originales. Se usaron niveles de ruido de 5%, 10% y 15%. Posteriormente, la red intentó recuperar el patrón original a partir de la versión ruidosa.

6. **Evaluación de la Memoria Hebbiana clásica.**  
   La Memoria Hebbiana utiliza una matriz de pesos calculada de forma similar, pero no realiza una dinámica iterativa de actualización hasta convergencia. En este caso, la respuesta se obtiene directamente a partir del patrón de entrada. Esta diferencia permite comparar el efecto de la dinámica recurrente propia de Hopfield.

7. **Comparación y análisis estadístico.**  
   Los resultados de ambas redes se compararon mediante métricas como tasa de recuperación correcta, similitud promedio entre patrón recuperado y patrón original, media y desviación estándar. Los resultados se organizaron en tablas y gráficas para facilitar la interpretación.

---

## 2. Gráficas de los resultados obtenidos para ambas redes

En esta sección deben integrarse las gráficas generadas a partir de los archivos CSV de resultados. Se recomienda guardar las figuras en la carpeta:

```text
results/figures/
```

Y las tablas numéricas en:

```text
results/tables/
```

### 2.1 Capacidad de almacenamiento por tamaño de imagen

La primera gráfica debe mostrar la tasa de recuperación correcta en función del número de patrones almacenados. En el eje X se coloca el número de letras almacenadas, de 1 a 26. En el eje Y se coloca la tasa de recuperación correcta.

Se recomienda generar una gráfica por tamaño de imagen:

- 7x7 pixeles
- 10x10 pixeles
- 15x15 pixeles

Cada gráfica debe incluir dos curvas:

- Red de Hopfield
- Memoria Hebbiana clásica

**Interpretación esperada:**  
A medida que aumenta el número de patrones almacenados, la tasa de recuperación tiende a disminuir. Esta degradación se debe a la interferencia entre patrones dentro de la matriz de pesos. Se espera que la Red de Hopfield tenga mejor recuperación que la Memoria Hebbiana clásica cuando existe ruido o cuando la recuperación requiere corrección iterativa.

Texto sugerido para el reporte:

> En las gráficas de capacidad de almacenamiento se observa que el desempeño de la red disminuye conforme aumenta el número de patrones almacenados. Esto ocurre porque los patrones empiezan a interferir entre sí dentro de la matriz de pesos. En tamaños pequeños, como 7x7, la degradación puede aparecer más rápido debido a que hay menos neuronas disponibles para representar cada letra. En tamaños mayores, como 15x15, la red cuenta con más dimensiones, por lo que las letras pueden diferenciarse mejor y la recuperación suele ser más estable.

### 2.2 Tolerancia al ruido

La segunda serie de gráficas debe mostrar la tasa de recuperación correcta para distintos niveles de ruido: 5%, 10% y 15%.

Se recomienda organizar las gráficas de la siguiente manera:

- Una gráfica por tamaño de imagen.
- En el eje X, el nivel de ruido.
- En el eje Y, la tasa de recuperación correcta.
- Dos curvas o barras: Hopfield y Hebbiana.

Texto sugerido para el reporte:

> En la evaluación con ruido se observa que la recuperación se vuelve más difícil conforme aumenta el porcentaje de bits alterados. La Red de Hopfield presenta una ventaja importante porque su dinámica iterativa permite corregir parcialmente los patrones ruidosos y llevarlos hacia un estado estable. En cambio, la Memoria Hebbiana clásica entrega una respuesta directa, por lo que tiene menor capacidad para corregir errores cuando el patrón de entrada está degradado.

### 2.3 Ejemplo visual de recuperación

Además de las gráficas numéricas, se recomienda incluir una comparación visual como la siguiente:

```text
Patrón original  ->  Patrón con ruido  ->  Patrón recuperado
```

Esta figura ayuda a mostrar de manera intuitiva el funcionamiento de la memoria asociativa. El patrón original corresponde a una letra limpia, el patrón con ruido tiene algunos bits invertidos y el patrón recuperado muestra la salida final de la red después de la actualización iterativa.

Texto sugerido para el reporte:

> La comparación visual permite observar cómo la Red de Hopfield intenta reconstruir el patrón original a partir de una versión alterada. Cuando el nivel de ruido es bajo o moderado, la red puede recuperar correctamente la letra almacenada. Sin embargo, cuando el ruido es demasiado alto o cuando hay demasiados patrones almacenados, la red puede converger a un patrón incorrecto o a un estado espurio.

---

## 3. Análisis comparativo entre la Red de Hopfield y la Memoria Hebbiana

La Red de Hopfield y la Memoria Hebbiana clásica comparten una base de aprendizaje similar, ya que ambas utilizan una matriz de pesos construida mediante una regla Hebbiana. Sin embargo, su principal diferencia está en el proceso de recuperación.

La Red de Hopfield funciona como una red recurrente. Esto significa que, después de recibir un patrón de entrada, actualiza sus neuronas de forma iterativa hasta llegar a un estado estable. Esta dinámica permite que la red corrija errores y complete patrones incompletos o ruidosos. Por esta razón, Hopfield puede comportarse como una memoria asociativa capaz de recuperar un patrón almacenado aun cuando la entrada no sea perfecta.

Por otro lado, la Memoria Hebbiana clásica sin dinámica realiza una recuperación directa. El patrón de entrada se multiplica por la matriz de pesos y se aplica una función de signo para obtener la salida. Al no existir un proceso iterativo de corrección, su desempeño suele ser más sensible al ruido y a la interferencia entre patrones.

En términos de capacidad de almacenamiento, ambas redes pueden presentar degradación cuando se almacenan demasiados patrones. Sin embargo, la Red de Hopfield suele mostrar una recuperación más robusta porque su dinámica permite reducir la energía del sistema y estabilizarse en patrones previamente aprendidos. Aun así, Hopfield también tiene limitaciones: si se almacenan demasiados patrones o si los patrones son muy similares entre sí, pueden aparecer mínimos espurios o recuperaciones incorrectas.

Tabla comparativa sugerida:

| Aspecto | Red de Hopfield | Memoria Hebbiana clásica |
|---|---|---|
| Tipo de recuperación | Iterativa hasta convergencia | Directa, sin dinámica |
| Manejo de ruido | Mayor tolerancia al ruido | Menor tolerancia al ruido |
| Matriz de pesos | Hebbiana simétrica, diagonal en cero | Hebbiana, usualmente sin dinámica posterior |
| Corrección de errores | Sí, mediante actualización recurrente | Limitada |
| Riesgo de estados espurios | Sí, especialmente con muchos patrones | No converge iterativamente, pero puede clasificar mal |
| Costo computacional | Mayor, por las iteraciones | Menor, por respuesta directa |

Texto sugerido para el reporte:

> La comparación muestra que la Red de Hopfield tiene una ventaja cuando los patrones de entrada contienen ruido, ya que su proceso de actualización iterativa permite acercar el estado de la red hacia un patrón almacenado. En contraste, la Memoria Hebbiana clásica depende más directamente de la calidad del patrón de entrada. Por ello, cuando el ruido aumenta, su rendimiento tiende a degradarse con mayor rapidez. No obstante, Hopfield requiere más costo computacional debido a sus iteraciones y puede presentar estados espurios cuando se supera su capacidad práctica de almacenamiento.

---

## 4. Conclusiones sobre la capacidad de almacenamiento

A partir de los experimentos realizados, se concluye que la capacidad de almacenamiento de una memoria asociativa depende del tamaño de los patrones, del número de patrones almacenados y del nivel de ruido presente en la entrada.

En el caso de los patrones de 7x7, la red cuenta con pocas neuronas, por lo que la representación de cada letra es más compacta y existe mayor probabilidad de interferencia entre patrones. Esto puede ocasionar que la recuperación comience a fallar con un menor número de letras almacenadas.

Para los patrones de 10x10 y 15x15, la red dispone de una mayor cantidad de neuronas, lo que permite representar con más detalle la forma de cada letra. Esto puede mejorar la separación entre patrones y aumentar la estabilidad de la recuperación. Sin embargo, el aumento de tamaño también implica una matriz de pesos más grande y, por lo tanto, mayor costo computacional.

La Red de Hopfield mostró ser más adecuada para tareas donde se requiere recuperar patrones a partir de entradas incompletas o ruidosas. Su dinámica iterativa permite corregir ciertos errores y estabilizar el estado de la red. En cambio, la Memoria Hebbiana clásica resulta más simple y rápida, pero menos robusta ante ruido.

Conclusión final sugerida:

> La Red de Hopfield es una alternativa más robusta que la Memoria Hebbiana clásica para recuperación de patrones con ruido, debido a su dinámica recurrente de actualización hasta convergencia. No obstante, su desempeño se degrada cuando se incrementa demasiado el número de patrones almacenados, especialmente en representaciones pequeñas como 7x7. Por ello, existe un compromiso entre capacidad, tamaño del patrón, tolerancia al ruido y costo computacional.

---

## 5. Fragmento listo para pegar en el reporte

El procedimiento experimental consistió en evaluar la capacidad de almacenamiento y recuperación de patrones en una Red de Hopfield utilizando letras del alfabeto codificadas en formato bipolar {-1,+1}. Se trabajó con tres tamaños de imagen: 7x7, 10x10 y 15x15 pixeles. Para cada tamaño, se almacenó progresivamente un conjunto de letras de la A a la Z y se midió la tasa de recuperación correcta. Posteriormente, se introdujeron niveles de ruido de 5%, 10% y 15% para analizar la tolerancia de la red ante patrones degradados.

La Red de Hopfield se entrenó mediante la regla de Hebb simétrica, eliminando las autoconexiones al colocar en cero la diagonal de la matriz de pesos. Durante la recuperación, la red actualizó iterativamente su estado hasta alcanzar convergencia o hasta cumplir un número máximo de iteraciones. Este comportamiento permite que la red funcione como una memoria asociativa, ya que puede recuperar un patrón previamente almacenado a partir de una versión parcial o ruidosa.

Los resultados deben compararse con una Memoria Hebbiana clásica sin dinámica. A diferencia de Hopfield, esta memoria produce una respuesta directa y no realiza iteraciones para corregir el patrón de entrada. Por ello, se espera que su desempeño sea menor cuando aumenta el nivel de ruido. La comparación entre ambas redes permite analizar la importancia de la dinámica recurrente en la recuperación de patrones.

En general, se espera que la tasa de recuperación disminuya conforme aumenta el número de patrones almacenados, debido a la interferencia entre memorias. También se espera que los tamaños de imagen mayores, especialmente 15x15, presenten un mejor desempeño que 7x7, ya que ofrecen una representación más rica de cada letra. Sin embargo, esto implica un mayor costo computacional, porque la matriz de pesos crece con el número de neuronas.

Finalmente, la Red de Hopfield representa una opción más robusta para recuperación de patrones ruidosos, mientras que la Memoria Hebbiana clásica destaca por su simplicidad. La elección entre ambas depende del objetivo del sistema: si se requiere rapidez y bajo costo, la Hebbiana puede ser suficiente; si se requiere corrección de errores y tolerancia al ruido, Hopfield ofrece mejores ventajas.
