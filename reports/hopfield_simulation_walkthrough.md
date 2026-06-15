# Simulación paso a paso: Red de Hopfield

Este documento muestra cómo se vería una simulación de la parte de Hopfield del proyecto.

La simulación está pensada para explicar el flujo completo antes de correr los experimentos finales:

1. Cargar patrones de letras en formato vectorial con valores `{-1, +1}`.
2. Entrenar una red de Hopfield con esos patrones.
3. Tomar una letra original.
4. Agregar ruido a un porcentaje de bits.
5. Intentar recuperar la letra original usando actualización iterativa.
6. Medir si la recuperación fue correcta.
7. Guardar resultados en archivos `.csv`.

> Nota: los resultados numéricos completos dependen del dataset `.npz` que esté en la carpeta `dataset/`. La simulación visual de este documento es un ejemplo ilustrativo con letras pequeñas de 7x7 para mostrar cómo se interpretan los resultados.

---

## 1. Archivos involucrados

La parte de Hopfield usa principalmente estos archivos:

```text
src/models/hopfield.py
src/experiments/hopfield_capacity.py
src/experiments/hopfield_noise.py
notebooks/hopfield_demo.ipynb
```

El archivo central es:

```text
src/models/hopfield.py
```

Ahí está la clase:

```python
HopfieldNetwork
```

con los métodos:

```python
train(patterns)
recall(pattern, max_iter=100)
predict(pattern)
energy(state)
```

---

## 2. Cómo se entrena la red

Supongamos que queremos almacenar tres letras:

```text
A, B, C
```

Cada letra se representa como un vector de valores `-1` y `+1`.

Por ejemplo, una letra de tamaño `7x7` tiene:

```text
7 * 7 = 49 neuronas
```

Entonces, si entrenamos con tres letras, la matriz de entrenamiento tiene esta forma:

```text
patterns.shape = (3, 49)
```

La red calcula los pesos usando la regla de Hebb:

```python
W = patterns.T @ patterns / n_neurons
```

Después se elimina la diagonal:

```python
np.fill_diagonal(W, 0.0)
```

Esto evita que una neurona se conecte consigo misma.

---

## 3. Ejemplo visual de una letra original

Para visualizar una letra en este documento usamos:

```text
█ = píxel activo de la letra, equivalente a +1
· = fondo, equivalente a -1
```

Ejemplo de letra `A` en 7x7:

```text
·······
···█···
··███··
··█·█··
··████·
·█···█·
·······
```

Esta es la letra limpia, sin ruido.

---

## 4. Agregar ruido

Si usamos ruido de `10%`, en una letra de 49 bits se alteran aproximadamente:

```text
49 * 0.10 = 4.9 ≈ 5 bits
```

Es decir, se seleccionan 5 posiciones aleatorias y se invierte su valor:

```python
+1 se convierte en -1
-1 se convierte en +1
```

Ejemplo de la misma letra `A` con 10% de ruido:

```text
····█··
···█···
··███··
█·█·█··
···███·
██···█·
······█
```

Visualmente ya no es exactamente la misma letra, pero todavía conserva parte de su estructura.

---

## 5. Recuperación con Hopfield

La red recibe el patrón ruidoso y lo actualiza iterativamente.

En cada iteración calcula:

```python
new_state = sign(W @ state)
```

La actualización se repite hasta que el patrón deja de cambiar o hasta alcanzar `max_iter`.

Un ejemplo de recuperación puede verse así:

```text
Original:
·······
···█···
··███··
··█·█··
··████·
·█···█·
·······

Ruidoso:
····█··
···█···
··███··
█·█·█··
···███·
██···█·
······█

Recuperado:
·······
···██··
··█·█··
··█·█··
··█·█··
···██··
·······
```

En este ejemplo, la red sí estabilizó un patrón, pero no recuperó exactamente la `A` original.

Eso es importante porque muestra una limitación normal de las redes de Hopfield: cuando el tamaño es pequeño o hay muchos patrones almacenados, la red puede converger a un estado estable incorrecto.

---

## 6. Energía de la red durante la recuperación

La energía se calcula con:

```python
E = -0.5 * state.T @ W @ state
```

En una red de Hopfield, la energía normalmente baja o se mantiene hasta llegar a un estado estable.

Ejemplo de energía durante la recuperación:

```text
Iteración 0: -19.9184
Iteración 1: -37.8776
Iteración 2: -47.3469
Iteración 3: -47.3469
```

Interpretación:

- En la iteración 0 está el patrón ruidoso inicial.
- La energía baja en las primeras actualizaciones.
- En la iteración 3 la energía ya no cambia.
- La red llegó a un estado estable.

En este ejemplo:

```text
Convergió: sí
Iteraciones: 3
Recuperación exacta: no
Similitud patrón ruidoso vs original: 0.8980
Similitud patrón recuperado vs original: 0.8367
```

Aunque convergió, no necesariamente recuperó bien la letra original.

---

## 7. Cómo se vería el experimento de capacidad

El archivo:

```text
src/experiments/hopfield_capacity.py
```

prueba cuántos patrones puede almacenar la red.

La lógica es:

```text
Para cada tamaño: 7x7, 10x10, 15x15
    Para n_patterns de 1 a 26
        Entrenar con las primeras n letras
        Intentar recuperar cada letra limpia
        Calcular accuracy
        Guardar resultados
```

Comando para correrlo:

```bash
python src/experiments/hopfield_capacity.py
```

Salida esperada:

```text
Saved: results/tables/hopfield_capacity.csv
```

Ejemplo de cómo se vería una tabla de resultados:

| size | n_neurons | n_patterns | stored_labels | correct | total | accuracy | mean_iterations |
|---:|---:|---:|---|---:|---:|---:|---:|
| 7 | 49 | 1 | A | 1 | 1 | 1.0000 | 1.0 |
| 7 | 49 | 2 | AB | 2 | 2 | 1.0000 | 1.0 |
| 7 | 49 | 3 | ABC | 0 | 3 | 0.0000 | 2.0 |

Interpretación:

- Con pocas letras, la red puede recuperar correctamente.
- Al aumentar patrones, especialmente en tamaños pequeños como 7x7, puede fallar pronto.
- Los tamaños 10x10 y 15x15 deberían tolerar más patrones porque tienen más neuronas.

---

## 8. Cómo se vería el experimento con ruido

El archivo:

```text
src/experiments/hopfield_noise.py
```

evalúa recuperación con ruido de:

```text
5%, 10%, 15%
```

La lógica es:

```text
Para cada tamaño: 7x7, 10x10, 15x15
    Para n_patterns de 1 a 26
        Entrenar Hopfield
        Para cada nivel de ruido: 5%, 10%, 15%
            Repetir varias veces
            Agregar ruido a cada patrón
            Recuperar con Hopfield
            Medir accuracy y similitud
            Guardar resultados
```

Comando para correrlo:

```bash
python src/experiments/hopfield_noise.py
```

Salida esperada:

```text
Saved: results/tables/hopfield_noise.csv
```

Ejemplo de tabla esperada:

| size | n_neurons | n_patterns | stored_labels | noise_percent | repetitions | correct | total_trials | accuracy | mean_noisy_similarity | mean_recovered_similarity |
|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 7 | 49 | 1 | A | 5 | 10 | 10 | 10 | 1.0000 | 0.9592 | 1.0000 |
| 7 | 49 | 1 | A | 10 | 10 | 9 | 10 | 0.9000 | 0.8980 | 0.9800 |
| 7 | 49 | 1 | A | 15 | 10 | 7 | 10 | 0.7000 | 0.8571 | 0.9300 |

Interpretación esperada:

- Con 5% de ruido, la recuperación debería ser mejor.
- Con 10% de ruido, pueden aparecer fallos.
- Con 15% de ruido, la recuperación suele degradarse más.
- Al aumentar el número de patrones almacenados, la accuracy normalmente baja.
- Los tamaños más grandes deberían resistir mejor porque tienen más bits para representar cada letra.

---

## 9. Cómo se vería el notebook demo

El notebook:

```text
notebooks/hopfield_demo.ipynb
```

debería mostrar tres imágenes:

```text
Original | Ruidoso | Recuperado
```

Ejemplo conceptual:

```text
Original              Ruidoso               Recuperado
·······               ····█··               ·······
···█···               ···█···               ···██··
··███··               ··███··               ··█·█··
··█·█··               █·█·█··               ··█·█··
··████·               ···███·               ··█·█··
·█···█·               ██···█·               ···██··
·······               ······█               ·······
```

La idea visual es que puedas comparar rápidamente:

- qué tan dañada quedó la imagen por el ruido,
- qué tanto corrigió Hopfield,
- si recuperó exactamente la letra original o no.

---

## 10. Pasos para correrlo en VS Code

Desde la terminal de VS Code, dentro del repositorio:

```bash
git pull origin main
```

Instala dependencias si todavía no lo hiciste:

```bash
pip install -r requirements.txt
```

Corre el experimento de capacidad:

```bash
python src/experiments/hopfield_capacity.py
```

Corre el experimento con ruido:

```bash
python src/experiments/hopfield_noise.py
```

Abre el notebook:

```text
notebooks/hopfield_demo.ipynb
```

Ejecuta las celdas en orden.

---

## 11. Qué revisar después de correr

Después de correr los scripts, deberían aparecer estos archivos:

```text
results/tables/hopfield_capacity.csv
results/tables/hopfield_noise.csv
```

En el reporte final, estos CSV servirán para hacer gráficas como:

```text
accuracy vs número de patrones almacenados
accuracy vs porcentaje de ruido
accuracy comparada entre 7x7, 10x10 y 15x15
```

---

## 12. Conclusión de esta simulación

Esta simulación muestra que Hopfield no solo memoriza patrones, sino que intenta llevar un patrón incompleto o ruidoso hacia un estado estable.

Sin embargo, un estado estable no siempre significa una recuperación correcta. La red puede fallar cuando:

- el tamaño de imagen es muy pequeño,
- hay demasiados patrones almacenados,
- los patrones son muy parecidos entre sí,
- el ruido es demasiado alto,
- la red converge a un mínimo espurio.

Por eso la práctica pide evaluar capacidad, tolerancia al ruido y resultados estadísticos en los tres tamaños de imagen.
